import logging

import torch
import torch_geometric.nn.conv

from torch_geometric.data import Data
from torch_geometric.typing import Adj
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, add_self_loops
from typing import List, Union
from torch_geometric.utils import to_undirected, degree
from torch_geometric.nn.conv import GCNConv, LEConv, PointNetConv, SplineConv

from .base.base import make_asynchronous, add_async_graph
from .base.utils import compute_edges, graph_changed_nodes, graph_new_nodes, cdist
from .flops import compute_flops_conv

from ..models.networks.my_conv import MyConv
from ..models.networks.my_fuse import MyConvBNReLU


def __graph_initialization(module, x: torch.Tensor, edge_index: Adj = None, edge_attr=None, **kwargs):
    pos = module.asy_pos
    if edge_attr is None:
        if edge_index is None:
            edge_index = compute_edges(module, pos=pos)
        if module.asy_edge_attributes is not None:
            attr_data = Data(pos=pos, edge_index=edge_index)
            edge_attr = module.asy_edge_attributes(attr_data).edge_attr



    if module.conv_type == 'spline':
        y = module.sync_forward(x, edge_index=edge_index, edge_attr=edge_attr)
    elif module.conv_type == 'gcn':
        y = module.sync_forward(x, edge_index=edge_index)
    elif module.conv_type == 'pointnet' or module.conv_type == 'my' or module.conv_type == 'fuse':
        y = module.sync_forward(x, pos=pos, edge_index=edge_index)
    else:
        if edge_attr is None:
            y = module.sync_forward(x, edge_index=edge_index)
        else:
            y = module.sync_forward(x, edge_index=edge_index, edge_attr=edge_attr)

    module.asy_graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)

    module.max_num_neighbors = 32
    module.num_old_edges = module.max_num_neighbors // 2
    module.available_neighbors = module.max_num_neighbors * torch.ones(module.sync_graph.num_nodes, device=pos.device)
    d = degree(edge_index[0, :], num_nodes=module.sync_graph.num_nodes)
    module.available_neighbors -= d


    # If required, compute the flops of the asynchronous update operation. Therefore, sum the flops for each node
    # update, as they highly depend on the number of neighbors of this node.
    if module.asy_flops_log is not None:
        flops = __compute_flops(module, idx_new=edge_index.unique().long(), idx_diff=[], edges=edge_index)
        module.asy_flops_log.append(flops)

    # If the layer is an initial layer, it will change the pos vector, when new events are added to the graph.
    # Therefore, we have to return the pos here as well.
    if module.asy_is_initial:
        module.asy_pass_attribute('asy_pos', pos)
    return module.asy_graph.y

#TODO: remove all asy_graph processing to the outside
def __graph_processing(module, x: torch.Tensor, edge_index = None, edge_attr: torch.Tensor = None, **kwargs):
    """Asynchronous graph update for graph convolutional layer.

    After the initialization of the graph, only the nodes (and their receptive field) have to updated which either
    have changed (different features) or have been added. Therefore, for updating the graph we have to first
    compute the set of "diff" and "new" nodes to then do the convolutional message passing on this subgraph,
    and add the resulting residuals to the graph.

    :param x: graph nodes features.
    """
    assert torch.nonzero(module.asy_graph.edge_index[0,:] <= module.asy_graph.num_nodes).numel()>0

    pos = module.asy_pos
    logging.debug(f"Input graph with x = {x.shape} and pos = {pos.shape}")
    logging.debug(f"Internal graph = {module.asy_graph}")
    if not module.asy_is_initial:  # deep layer
        x_new, idx_new = graph_new_nodes(module, x=x)
        pos_new = pos[idx_new, :]
        _, idx_diff = graph_changed_nodes(module, x=x)
        if idx_diff.numel() > 0:
            idx_diff, _, _, _ = k_hop_subgraph(idx_diff, num_hops=1, edge_index=module.asy_graph.edge_index,
                                               num_nodes=module.asy_graph.num_nodes + len(idx_new))
        x_all = x
        pos_all = pos
    else:  # initial layer
        num_prev_nodes = module.asy_graph.num_nodes
        x_new, pos_new = x, pos
        idx_new = torch.arange(num_prev_nodes, num_prev_nodes + x.size()[0], device=x.device)
        idx_diff = torch.tensor([], device=x.device, dtype=torch.long)
        x_all = torch.cat([module.asy_graph.x, x_new], dim=0)
        pos_all = torch.cat([module.asy_graph.pos, pos_new], dim=0)

    logging.debug(f"Subgraph contains {idx_new.numel()} new and {idx_diff.numel()} diff nodes")
    # node_distance = torch.cdist(pos_all, pos_new)  # for PyTorch >= 1.13.0
    # node_distance = (pos_all-pos_new).pow(2).sum(1).sqrt().view(-1,1)
    node_distance = cdist(pos_all, pos_new, p=1).view(-1,1)
    eps = 1e-5
    connected_node_mask = node_distance <= (module.asy_radius + eps)
    connected_node_mask[-1, :] = False  # remove self loop

    full_neighbors_idx = (module.available_neighbors[:pos_all.shape[0]] <= 0).nonzero()
    connected_node_mask[full_neighbors_idx, :] = False # if full, hidden this node

    idx_new_neigh = torch.unique(torch.nonzero(connected_node_mask)[:, 0])
    if idx_new_neigh.numel() > module.num_old_edges:
            idx_new_neigh = idx_new_neigh[:module.num_old_edges] # clamp, reserve some edges for future nodes; for now, first connect older nodes, not nearer nodes
    idx_update = torch.cat([idx_new, idx_new_neigh, idx_diff])
    # print(f'idx_new:{idx_new}')
    _, edges_connected, _, connected_edges_mask = k_hop_subgraph(idx_update, num_hops=1,
                                                                 edge_index=module.asy_graph.edge_index,
                                                                 num_nodes=pos_all.shape[0])

    edge_attr = None
    if idx_new.numel() > 0:
        # edges_new = torch.nonzero(connected_node_mask).T
        # edges_new[1, :] = idx_new[edges_new[1, :]]
        # edges_new_inv = torch.stack([edges_new[1, :], edges_new[0, :]], dim=0)
        # edges_new = torch.cat([edges_new, edges_new_inv], dim=1)
        # edges_new = torch.unique(edges_new, dim=1)  # rm doubled edges from concatenating the inverse
        # # torch.unique cannot keep 'size' info when 'edges_new' is a tensor([]). It's a bug and will be fixed in a future pytorch version
        # if edges_new.shape[0] != 2:
        #     edges_new = edges_new.view(2,0)

        # create connected nodes idx
        idx_src = torch.nonzero(connected_node_mask)[:, 0].to(torch.long)
        idx_src = idx_src.to(x.device)
        if idx_src.numel() > module.num_old_edges:
            idx_src = idx_src[:module.num_old_edges] # clamp, reserve some edges for future nodes; for now, first connect older nodes, not nearer nodes
        idx_dst = (idx_new * torch.ones_like(idx_src, device=x.device)).to(torch.long)

        # when connected, -1 available neighbors for every 2 nodes of an edge
        module.available_neighbors[idx_src] -= 1
        module.available_neighbors[idx_dst] -= idx_dst.numel()

        # create edges
        edges_new = torch.cat([idx_src.unsqueeze(0), idx_dst.unsqueeze(0)], dim=0)

        edges_new = to_undirected(edges_new)
        # edges_new,_ = remove_self_loops(edges_new)
        edge_index = torch.cat([edges_connected, edges_new], dim=1)


        if module.asy_edge_attributes is not None:
            graph_new = Data(x=x_all, pos=pos_all, edge_index=edges_new)
            edge_attr_new = module.asy_edge_attributes(graph_new).edge_attr
            edge_attr_connected = module.asy_graph.edge_attr[connected_edges_mask, :]
            edge_attr = torch.cat([edge_attr_connected, edge_attr_new])
    else:
        edge_index = edges_connected
        if module.asy_edge_attributes is not None:
            edge_attr = module.asy_graph.edge_attr[connected_edges_mask, :]

    need_self_loops = getattr(module, 'add_self_loops', False)
    if need_self_loops:
        edge_index,_ = add_self_loops(edge_index)

    out_channels = module.asy_graph.y.size()[-1]
    y = torch.cat([module.asy_graph.y.clone(), torch.zeros(x_new.size()[0], out_channels, device=x.device)])
    if edge_index.numel() > 0:
        if module.conv_type == 'spline':
            # # original:
            # x_j = x_all[edge_index[0, :], :]
            # if edge_attr is not None:
            #     phi = module.message(x_j, edge_attr=edge_attr)
            # else:
            #     x_j = torch.matmul(x_j, module.weight)
            #     phi = module.message(x_j, edge_weight=None)

            # # Use the internal message passing for feature aggregation.
            # y_update = module.aggregate(phi, index=edge_index[1, :], ptr=None, dim_size=x_all.size()[0])
            y_update_test = module.propagate(edge_index, x=x_all, edge_attr=edge_attr, size=None)
            y_update = y_update_test
        elif module.conv_type == 'pointnet' or module.conv_type == 'my':
            # # pointnet
            # x_j = x_all[edge_index[0, :], :]
            # x_i = x_all[edge_index[1, :], :]
            # pos_j = pos_all[edge_index[0, :], :]
            # pos_i = pos_all[edge_index[1, :], :]
            # phi = module.message(x_j, pos_i=pos_i, pos_j=pos_j)

            # # Use the internal message passing for feature aggregation.
            # y_update = module.aggregate(phi, index=edge_index[1, :], ptr=None, dim_size=x_all.size()[0])
            y_update_test = module.propagate(edge_index, x=x_all, pos=pos_all, size=None)
            y_update = y_update_test
        elif module.conv_type == 'fuse':
            # y_update_test = module.propagate(edge_index, x=x_all, pos=pos_all, size=None)
            from tqdm import tqdm
            tprint = tqdm.write
            # tprint(f'now in fuse')
            y_update_test = module.sync_forward(x=x_all, pos=pos_all, edge_index=edge_index)
            y_update = y_update_test
        elif module.conv_type == 'gcn':
            # gcn
            x_all_new = module.lin(x_all)
            x_j = x_all_new[edge_index[0, :], :]
            x_i = x_all_new[edge_index[1, :], :]
            phi = module.message(x_j, edge_weight=None)

            # Use the internal message passing for feature aggregation.
            y_update = module.aggregate(phi, index=edge_index[1, :], ptr=None, dim_size=x_all_new.size()[0])
            y_update_test = module.propagate(edge_index, x=x_all_new, edge_weight=None, size=None)

        # all_close = torch.allclose(y_update, y_update_test, atol=1e-5)
        # if not all_close:
        #     where_close = torch.isclose(y_update, y_update_test)
        #     where_c = torch.nonzero(~where_close)


        # Concat old and updated feature for output feature vector.
        y[idx_update] = y_update_test[idx_update]
    logging.debug(f"Updated {idx_update.numel()} nodes in asy. graph of module {module}")


    # If required, compute the flops of the asynchronous update operation. Therefore, sum the flops for each node
    # update, as they highly depend on the number of neighbors of this node.
    if module.asy_flops_log is not None:
        flops = __compute_flops(module, idx_new=idx_new, idx_diff=idx_diff, edges=edge_index)
        module.asy_flops_log.append(flops)

    # If the layer is an initial layer, it will change the pos vector, when new events are added to the graph.
    # Therefore, we have to return the pos here as well.
    if module.asy_is_initial:
        module.asy_pass_attribute('asy_pos', pos_all)

    # updating
    if idx_new.numel() > 0:
        module.asy_graph.x = x_all
        module.asy_graph.pos = pos_all
        module.asy_graph.edge_index = torch.cat([module.asy_graph.edge_index, edges_new], dim=1)
        module.asy_graph.edge_attr = torch.cat([module.asy_graph.edge_attr, edge_attr_new])
        module.asy_graph.y = y

    return y


def __compute_flops(module, idx_new: Union[torch.LongTensor, List[int]], idx_diff: Union[torch.LongTensor, List[int]],
                    edges: torch.LongTensor) -> int:
    if not isinstance(idx_new, list):
        idx_new = idx_new.detach().cpu().numpy().tolist()
    if not isinstance(idx_diff, list):
        idx_diff = idx_diff.detach().cpu().numpy().tolist()
    return compute_flops_conv(module, idx_new=idx_new, idx_diff=idx_diff, edges=edges)


def __check_support(module) -> bool:
    if isinstance(module, torch_geometric.nn.conv.GCNConv):
        if module.normalize is True:
            raise NotImplementedError("GCNConvs with normalization are not yet supported!")
            # pass
    elif isinstance(module, torch_geometric.nn.conv.SplineConv):
        if module.bias is not None:
            raise NotImplementedError("SplineConvs with bias are not yet supported!")
            # pass
        # if module.root is not None:
        # if hasattr(module, 'root') and module.root is not None:
        if module.root_weight is True:
            raise NotImplementedError("SplineConvs with root weight are not yet supported!")
    return True


def make_conv_asynchronous(module, r: float, edge_attributes=None, is_initial: bool = False,
                           log_flops: bool = False, log_runtime: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for graph convolutional layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a convolutional layer can be converted by, for example:

    ```
    module = GCNConv(1, 2)
    module = make_conv_asynchronous(module)
    ```

    :param module: convolutional module to transform.
    :param r: update radius around new events.
    :param edge_attributes: function for computing edge attributes (default = None).
    :param is_initial: layer initial layer of sequential or deeper (default = False).
    :param log_flops: log flops of asynchronous update.
    :param log_runtime: log runtime of asynchronous update.
    """
    assert __check_support(module)

    module = add_async_graph(module, r=r, log_flops=log_flops, log_runtime=log_runtime)
    module.asy_pos = None
    module.asy_is_initial = is_initial
    module.asy_edge_attributes = edge_attributes
    module.sync_forward = module.forward

    module.sync_graph = None #TODO: for debug

    if isinstance(module, SplineConv): module.conv_type = 'spline'
    elif isinstance(module, GCNConv): module.conv_type = 'gcn'
    elif isinstance(module, PointNetConv): module.conv_type = 'pointnet'
    elif isinstance(module, LEConv): module.conv_type = 'le'
    elif isinstance(module, MyConv): module.conv_type = 'my'
    elif isinstance(module, MyConvBNReLU): module.conv_type = 'fuse'
    else: module.conv_type = 'other'

    return make_asynchronous(module, __graph_initialization, __graph_processing)
