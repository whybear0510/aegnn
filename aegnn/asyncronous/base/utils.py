import torch

from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import to_undirected, degree
from typing import Tuple
from tqdm import tqdm, trange
from time import time


def graph_changed_nodes(module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
    num_prev_nodes = module.asy_graph.num_nodes
    x_graph = x[:num_prev_nodes]
    different_node_idx = (~torch.isclose(x_graph, module.asy_graph.x)).long()
    different_node_idx = torch.nonzero(torch.sum(different_node_idx, dim=1))[:, 0]
    return x_graph, different_node_idx


def graph_new_nodes(module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
    num_prev_nodes = module.asy_graph.num_nodes
    assert x.size()[0] >= num_prev_nodes, "node deletion is not supported"
    x_graph = x[num_prev_nodes:]
    new_node_idx = torch.arange(num_prev_nodes, x.size()[0], device=x.device, dtype=torch.long).long()
    return x_graph, new_node_idx


def compute_edges(module, pos: torch.Tensor) -> torch.LongTensor:
    return radius_graph(pos, r=module.asy_radius, max_num_neighbors=pos.size()[0])

def pos_dist(pos: torch.Tensor) -> torch.Tensor:
    num_nodes = pos.shape[0]
    node_distance = []
    for i in range(num_nodes):
        d = (pos-pos[i, :]).pow(2).sum(1).sqrt().view(1,-1)
        node_distance.append(d)
    node_distances = torch.cat(node_distance, dim=0)
    return node_distances

@torch.no_grad()
def causal_radius_graph(data_pos: torch.Tensor, r: float, max_num_neighbors: int = 32, reserve_future_edges: bool = True) -> torch.LongTensor:
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    pos = data_pos.clone().detach()
    pos = pos.to('cuda')

    device = pos.device
    num_nodes = pos.shape[0]

    # allowed edges to connect older nodes; reserve some edges for future nodes
    if reserve_future_edges:
        num_old_edges = max_num_neighbors // 2
    else:
        num_old_edges = max_num_neighbors

    # set neighbors counter
    available_neighbors = max_num_neighbors * torch.ones(num_nodes, device=device)

    # calculate collection distance
    # dist_table = torch.cdist(pos, pos, p=2) # for PyTorch >= 1.13.0
    dist_table = pos_dist(pos)
    connection_mask = dist_table <= r
    connection_mask = connection_mask.to(device)

    edges_new_list = []
    for i in range(num_nodes):
        # check available neighbors
        full_neighbors_idx = (available_neighbors <= 0).nonzero()
        connection_mask[full_neighbors_idx, :] = False # if full, hidden this node

        # connection mask for the i-th node
        node_connection_mask = connection_mask[:i, i]  # ":i" can remove self-loop

        # create connected nodes idx
        idx_src = torch.nonzero(node_connection_mask).to(torch.long)
        idx_src = idx_src.to(device)
        if idx_src.numel() > num_old_edges:
            idx_src = idx_src[:num_old_edges] # clamp, reserve some edges for future nodes; for now, first connect older nodes, not nearer nodes
        idx_dst = (i * torch.ones_like(idx_src, device=device)).to(torch.long)

        # when connected, -1 available neighbors for every 2 nodes of an edge
        available_neighbors[idx_src] -= 1
        available_neighbors[idx_dst] -= idx_dst.numel()

        # create edges
        edge_new = torch.cat([idx_src, idx_dst], dim=1).T
        edge_new = to_undirected(edge_new)
        edges_new_list.append(edge_new)

    edges_new = torch.cat(edges_new_list, dim=1)
    # edges_new = to_undirected(edges_new)
    edge_index = edges_new.detach().clone().to('cpu')

    return edge_index
