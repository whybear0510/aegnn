import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from tqdm import tqdm
import torch
import aegnn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree, remove_self_loops

from aegnn.asyncronous.base.utils import hugnet_graph
from aegnn.models.networks.my_fuse import MyConvBNReLU
from aegnn.asyncronous import make_model_asynchronous
from aegnn.asyncronous import MaxPoolingX

import pytorch_lightning as pl

device = torch.device('cuda')
pl.seed_everything(123)
num_nodes = 20000
r = 3.0
max_num_neighbors = 32
input_length = 120

data_pos = torch.rand(num_nodes,3)
data_pos[:, 2],_ = data_pos[:, 2].sort()
data_pos[:, 2] *= 1e-6
data_pos[:, :2] = (data_pos[:, :2]*input_length).round()
# print(data_pos)

edge_index = hugnet_graph(data_pos, r=r, max_num_neighbors=max_num_neighbors)
# print(edge_index)

x = (torch.rand(num_nodes,1)).round()
# print(x)

data = Data(x=x, pos=data_pos, edge_index=edge_index)

data = data.to(device)
torch.cuda.empty_cache()

class Test(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = device

        self.net1 = MyConvBNReLU(1,16)
        self.net1 = self.net1.to(self.device)
        self.net1.eval()

        # self.net1.local_nn.weight = torch.nn.Parameter(torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,1]], dtype=torch.float))

        self.net1.bn.module.running_mean = (torch.rand_like(self.net1.bn.module.running_mean, device=self.device))
        self.net1.bn.module.running_var = (torch.rand_like(self.net1.bn.module.running_var, device=self.device))
        self.net1.bn.module.weight = torch.nn.Parameter(torch.rand_like(self.net1.bn.module.weight, device=self.device))
        self.net1.bn.module.bias = torch.nn.Parameter(torch.rand_like(self.net1.bn.module.bias, device=self.device))

        # self.net1.to_fused()


        self.net2 = MyConvBNReLU(16,32)
        self.net2 = self.net2.to(self.device)
        self.net2.eval()

        self.net2.bn.module.running_mean = (torch.rand_like(self.net2.bn.module.running_mean, device=self.device))
        self.net2.bn.module.running_var = (torch.rand_like(self.net2.bn.module.running_var, device=self.device))
        self.net2.bn.module.weight = torch.nn.Parameter(torch.rand_like(self.net2.bn.module.weight, device=self.device))
        self.net2.bn.module.bias = torch.nn.Parameter(torch.rand_like(self.net2.bn.module.bias, device=self.device))

        # self.net2.to_fused()

        num_grids = 8*8
        pooling_dm_dims = torch.tensor([16.,16.], device=self.device)
        input_shape = torch.tensor([input_length, input_length], dtype=torch.float, device=self.device)
        self.pool = MaxPoolingX(pooling_dm_dims, size=num_grids, img_shape=input_shape)

    def forward(self, data):
        out1 = self.net1(x=data.x, pos=data.pos, edge_index=data.edge_index)
        out2 = self.net2(x=out1, pos=data.pos, edge_index=data.edge_index)
        out3,_ = self.pool(out2, pos=data.pos[:, :2], batch=None)
        return out3

model = Test()
model = model.to(device)
with torch.no_grad():
    sync_out = model(data)

    async_model = make_model_asynchronous(model, r=r, max_num_neighbors=max_num_neighbors)

    async_outs = []

    for idx in tqdm(range(data.num_nodes)):
        torch.cuda.empty_cache()
        x_new = data.x[idx, :].view(1, -1)
        pos_new = data.pos[idx, :3].view(1, -1)
        event_new = Data(x=x_new, pos=pos_new, batch=torch.zeros(1, dtype=torch.long))
        event_new = event_new.to(device)
        async_out = async_model(event_new)
        async_outs.append(async_out)
    # tot_async_out = torch.cat(async_outs)
    tot_async_out = async_outs[-1]
    same = torch.allclose(tot_async_out, sync_out, atol=1e-3)
    print(same)
    where = torch.nonzero(~torch.isclose(tot_async_out, sync_out, atol=1e-3))

    async_edge = async_model.asy_graph.edge_index
    sync_edge = data.edge_index
    async_edge,_ = remove_self_loops(async_edge)
    sync_edge,_ = remove_self_loops(sync_edge)
    same_e = torch.allclose(async_edge, sync_edge)
    print(same_e)