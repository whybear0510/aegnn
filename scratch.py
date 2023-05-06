import torch
import aegnn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from aegnn.asyncronous.base.utils import hugnet_graph
from aegnn.models.networks.my_fuse import MyConvBNReLU
from aegnn.asyncronous import make_model_asynchronous
import pytorch_lightning as pl
# pl.seed_everything(123)
num_nodes = 100
r = 3.0
max_num_neighbors = 32

data_pos = torch.rand(num_nodes,3) *128
data_pos[:, :2] = data_pos[:, :2].round()
data_pos[:, 2],_ = data_pos[:, 2].sort()
data_pos[:, 2] *= 1e-6
# print(data_pos)

edge_index = hugnet_graph(data_pos, r=r, max_num_neighbors=max_num_neighbors)
# print(edge_index)

x = (torch.rand(num_nodes,1)).round()
# print(x)

data = Data(x=x, pos=data_pos, edge_index=edge_index)

class Test(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cpu')

        self.net1 = MyConvBNReLU(1,16)
        self.net1.eval()

        # self.net1.local_nn.weight = torch.nn.Parameter(torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,1]], dtype=torch.float))

        self.net1.bn.module.running_mean = (torch.rand_like(self.net1.bn.module.running_mean))
        self.net1.bn.module.running_var = (torch.rand_like(self.net1.bn.module.running_var))
        self.net1.bn.module.weight = torch.nn.Parameter(torch.rand_like(self.net1.bn.module.weight))
        self.net1.bn.module.bias = torch.nn.Parameter(torch.rand_like(self.net1.bn.module.bias))

        self.net1.to_fused()


        self.net2 = MyConvBNReLU(16,32)
        self.net2.eval()

        self.net2.bn.module.running_mean = (torch.rand_like(self.net2.bn.module.running_mean))
        self.net2.bn.module.running_var = (torch.rand_like(self.net2.bn.module.running_var))
        self.net2.bn.module.weight = torch.nn.Parameter(torch.rand_like(self.net2.bn.module.weight))
        self.net2.bn.module.bias = torch.nn.Parameter(torch.rand_like(self.net2.bn.module.bias))

        self.net2.to_fused()

    def forward(self, data):
        out1 = self.net1(x=data.x, pos=data.pos, edge_index=data.edge_index)
        out2 = self.net2(x=out1, pos=data.pos, edge_index=data.edge_index)
        return out2

model = Test()
with torch.no_grad():
    sync_out = model(data)

async_model = make_model_asynchronous(model, r=r, max_num_neighbors=max_num_neighbors)

from tqdm import tqdm
for idx in tqdm(range(data.num_nodes)):

    x_new = data.x[idx, :].view(1, -1)
    pos_new = data.pos[idx, :3].view(1, -1)
    event_new = Data(x=x_new, pos=pos_new, batch=torch.zeros(1, dtype=torch.long))
    async_out = async_model(event_new)