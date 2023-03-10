import torch
import aegnn
import torch_geometric
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SplineConv, BatchNorm
from torch.nn.functional import elu
import pytorch_lightning as pl
import networkx as nx
from torch_geometric.utils import to_networkx
from aegnn.models.layer import MaxPooling, MaxPoolingX
from aegnn.asyncronous.base.utils import causal_radius_graph
from torch_geometric.utils import to_undirected, degree
from tqdm import tqdm, trange

pl.seed_everything(12345)
device = torch.device('cuda')

def sample_initial_data(sample, num_events: int, radius: float, edge_attr, max_num_neighbors: int):
    data = Data(x=sample.x[:num_events], pos=sample.pos[:num_events])
    subset = torch.arange(num_events)
    data.edge_index, data.edge_attr = torch_geometric.utils.subgraph(subset, sample.edge_index, sample.edge_attr)
    nxt_event_idx = num_events
    return data, nxt_event_idx

def sample_new_data(sample, nxt_event_idx):
    x_new = sample.x[nxt_event_idx, :].view(1, -1)
    pos_new = sample.pos[nxt_event_idx, :3].view(1, -1)  #TODO: :2 ? no time?
    event_new = Data(x=x_new, pos=pos_new, batch=torch.zeros(1, dtype=torch.long))
    nxt_event_idx += 1
    return event_new, nxt_event_idx

# def network
class net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = SplineConv(1, 2, dim=3, kernel_size=2, bias=False, root_weight=False)
        self.conv2 = SplineConv(2, 4, dim=3, kernel_size=2, bias=False, root_weight=False)
        # self.conv1 = GCNConv(1, 2)
        # self.conv2 = GCNConv(2, 4)
        self.norm1 = BatchNorm(in_channels=2)
        self.norm2 = BatchNorm(in_channels=4)
        self.act = elu

        self.input_shape = torch.tensor([100,100,3]).to(device)
        grid_div = 4
        num_grids = grid_div*grid_div
        pooling_dm_dims = torch.div(self.input_shape[:2], grid_div)
        self.pool7 = MaxPoolingX(pooling_dm_dims, size=num_grids, img_shape=self.input_shape[:2])
        self.fc = Linear(4 * num_grids, out_features=4, bias=False)

    def forward(self, data):
        data.x = self.conv1(data.x, data.edge_index, data.edge_attr)
        # print(f'after conv1:\n{data.x}\n')

        data.x = self.act(data.x)
        # print(f'after elu1:\n{data.x}\n')

        data.x = self.norm1(data.x)
        # print(f'after norm1:\n{data.x}')

        data.x = self.conv2(data.x, data.edge_index, data.edge_attr)
        # print(f'after conv2:\n{data.x}\n')

        data.x = self.norm2(data.x)
        # print(f'after norm2:\n{data.x}')

        # TODO: sudo-async, only for debug
        if hasattr(self.conv1, 'asy_graph'):
            data.pos = self.conv1.asy_graph.pos
            data.batch = None
        x = self.pool7(data.x, pos=data.pos[:,:2], batch=data.batch)
        # print(f'after pool:\n{x}\n')
        x = x.view(-1, self.fc.in_features) # x.shape = [batch_size, num_grids*num_last_hidden_features]
        output = self.fc(x)
        # print(f'after fc:\n{output}\n')

        return output

module = net()
module = module.to(device)
module.eval()

nn_layers = module._modules
for key, nn in nn_layers.items():
    nn_class_name = nn.__class__.__name__

    if isinstance(nn, BatchNorm):
        nn_layers[key].module.running_mean = 2*torch.ones_like(nn_layers[key].module.running_mean, device=device)
        nn_layers[key].module.running_var  = torch.ones_like(nn_layers[key].module.running_var, device=device)
        nn_layers[key].module.weight = torch.nn.Parameter(torch.ones_like(nn_layers[key].module.weight, device=device))
        nn_layers[key].module.bias = torch.nn.Parameter(torch.zeros_like(nn_layers[key].module.bias, device=device))
        nn_layers[key].module.eps = 0.0
        # pass

# pl_module = torch.load('/users/yyang22/thesis/aegnn_project/aegnn_results/training_results/checkpoints/ncars/recognition/20230227151127/epoch=71-step=14615.pt')
# module = pl_module.model
# module = module.to(device)
# module.eval()

attr_func = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)

with torch.no_grad():



    # def graph for sync

    # graph = Data(
    #     x = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device).view(-1,1),
    #     pos = torch.tensor([[2,3,1e-9],[1,2,2e-9],[1,1,3e-9],[3,1,4e-9],[3,2,5e-9],[5,3,6e-9],[5,2,7e-9],[4,2,8e-9]], device=device),
    #     edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,0,5,6,4,6,7,7],
    #                                [1,0,2,1,3,2,4,3,0,4,6,5,7,7,4,6]], device=device, dtype=torch.long))
    radius = 3.0
    # num_nodes = 2122
    num_nodes = 2222

    x_n = torch.round(torch.rand(10000, device=device), decimals=0).view(-1,1)
    pos_n = torch.rand(10000,3)*98.0
    pos_n[:,:2] = torch.round(pos_n[:,:2])
    pos_n[:,2] *= 1e-10
    pos_n = pos_n[torch.argsort(pos_n[:,2]).squeeze(),:]
    pos_n = pos_n.to(device)

    # x_n = torch.ones(10000, device=device).view(-1,1)
    # pos_n = torch.ones(10000,3)
    # # pos_n[:,:2] = torch.round(pos_n[:,:2])
    # pos_n[:,2] *= 1e-10
    # # pos_n = pos_n[torch.argsort(pos_n[:,2]).squeeze(),:]
    # pos_n = pos_n.to(device)

    graph = Data(
        x = x_n[:num_nodes,:],
        pos = pos_n[:num_nodes,:]
    )
    graph.edge_index = causal_radius_graph(graph.pos, r=radius, max_num_neighbors=32).to(device)
    graph = attr_func(graph)
    # print(graph.edge_index)
    graph_copy = graph.clone().detach().to(device)

    print('sync, graph:')
    sync_o = module.forward(graph)
    print(sync_o)
    # print(f'sync graph.num_nodes = {graph.num_nodes}')


    # # def graph_init, node_x for AEGNN
    # x = torch.tensor([1.0, 1.0], device=device).view(-1,1)
    # edge_index = torch.tensor([[0,1],
    #                            [1,0]], device=device, dtype=torch.long)
    # pos = torch.tensor([[0,0,0e-9],[0,1,1e-9]], device=device)

    # graph_init = Data(x=x, pos=pos, edge_index=edge_index)
    # graph_init = attr_func(graph_init)

    # node_1 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,2,2e-9]], device=device))
    # node_2 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,3,3e-9]], device=device))
    # node_3 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,4,4e-9]], device=device))
    # node_4 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,5,5e-9]], device=device))
    # node_5 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,6,6e-9]], device=device))
    # node_6 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,7,7e-9]], device=device))
    # node_7 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,8,8e-9]], device=device))
    # node_8 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,9,9e-9]], device=device))
    # node_9 = Data(x=torch.tensor([[1.0]], device=device), pos=torch.tensor([[0,10,1e-8]], device=device))

    # async model
    module = aegnn.asyncronous.make_model_asynchronous(module, radius, (10,10), attr_func)
    module.eval()
    module = aegnn.asyncronous.reset_async_module(module)
    aegnn.asyncronous.register_sync_graph(module, graph_copy)


    # print('async, init:')
    # asy_o = asy_aegnn.forward(graph_init)
    # print(asy_o)
    # print('async, add node 1:')
    # asy_o = asy_aegnn.forward(node_1)
    # print(asy_o)
    # print('async, add node 2:')
    # asy_o = asy_aegnn.forward(node_2)
    # print(asy_o)
    # print('async, add node 3:')
    # asy_o = asy_aegnn.forward(node_3)
    # print(asy_o)

    # events_initial, nxt_event_idx = sample_initial_data(graph_copy, 5, 1.1, attr_func, 32)
    # event_1, nxt_event_idx = sample_new_data(graph_copy, nxt_event_idx)
    # event_2, nxt_event_idx = sample_new_data(graph_copy, nxt_event_idx)
    # event_3, nxt_event_idx = sample_new_data(graph_copy, nxt_event_idx)
    # # print(events_initial.pos,event_1.pos,event_2.pos,event_3.pos)

    # print('async, init:')
    # asy_o = asy_aegnn.forward(events_initial)
    # # print(asy_o)
    # print('async, add event 1:')
    # asy_o = asy_aegnn.forward(event_1)
    # # print(asy_o)
    # print('async, add event 2:')
    # asy_o = asy_aegnn.forward(event_2)
    # # print(asy_o)
    # print('async, add event 3:')
    # asy_o = asy_aegnn.forward(event_3)
    # print(asy_o)


    events_initial, nxt_event_idx = sample_initial_data(graph_copy, 2, radius, attr_func, 32)
    while nxt_event_idx < num_nodes:
        if events_initial.edge_index.numel() > 0:
            break
        else:
            events_initial, nxt_event_idx = sample_initial_data(graph_copy, nxt_event_idx+1, radius, attr_func, 32)
    print(f'1st edge started from node {nxt_event_idx}')
    # events_initial, nxt_event_idx = sample_initial_data(graph_copy, num_nodes-80, radius, attr_func, 32)
    asy_o = module.forward(events_initial)

    # for i in trange(num_nodes-2):
    #     if i==0:
    #         # print('async, init:')
    #         asy_o = asy_aegnn.forward(events_initial)
    #     else:
    #         # print(f'async, add event {i}:')
    #         event, nxt_event_idx = sample_new_data(graph_copy, nxt_event_idx)
    #         asy_o = asy_aegnn.forward(event)
    with tqdm(total=(num_nodes-nxt_event_idx), leave=False, desc='Events') as pbar:
        while nxt_event_idx < num_nodes:
            torch.cuda.empty_cache()
            event_new, nxt_event_idx = sample_new_data(graph_copy, nxt_event_idx)
            event_new = event_new.to(device)
            asy_o = module.forward(event_new)
            # print(f'\nasy output:{asy_o}')
            all_close = torch.allclose(module.conv1.available_neighbors, module.conv2.available_neighbors)
            if not all_close:
                err_idx = torch.nonzero(~torch.isclose(module.conv1.available_neighbors, module.conv2.available_neighbors))
                print(f'not all close at idx {err_idx}')
            pbar.update(1)
    print(f'\nasy output:{asy_o}')
    print(f'asy o == sync o ? :{torch.allclose(asy_o, sync_o)}')

    aegnn_graph = module.conv1.asy_graph
    # aegnn_graph.edge_index = to_undirected(aegnn_graph.edge_index)
    # for i in range(graph.edge_index.shape[1]):
    #     test_flag = torch.allclose(aegnn_graph.edge_index[:,:i+1], graph.edge_index[:,:i+1])
    #     if not test_flag: print(i)
    print(f'asy graph == sync graph ? :{torch.allclose(aegnn_graph.edge_index, graph.edge_index)}')
    # print(f'async edge: \n{graph_copy.edge_index}')
    # print(f'conv2.graph_copy.num_nodes: \n{graph_copy.num_nodes}')
