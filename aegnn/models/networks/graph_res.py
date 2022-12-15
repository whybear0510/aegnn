import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch.nn.functional import relu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.conv import GCNConv, LEConv, PointNetConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX


class GraphRes(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False, act: str = 'elu', grid_div: int = 4):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.pooling_size = torch.tensor(pooling_size, device=device)
        self.input_shape = input_shape.to(device)

        if act == 'relu':
            self.act = relu
        elif act == 'elu':
            self.act = elu
        else:
            raise ValueError('Unsupported activation function')

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        self.type = 'GCNConv'
        # self.type = 'LEConv'
        # self.type = 'PointNetConv'
        # self.type = 'SplineConv'

        if self.type == 'SplineConv':
            self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        elif self.type == 'GCNConv':
            self.conv1 = GCNConv(n[0], n[1])
            self.conv2 = GCNConv(n[1], n[2])
            self.conv3 = GCNConv(n[2], n[3])
            self.conv4 = GCNConv(n[3], n[4])
            self.conv5 = GCNConv(n[4], n[5])
            self.conv6 = GCNConv(n[5], n[6])
            self.conv7 = GCNConv(n[6], n[7])
        elif self.type == 'LEConv':
            self.conv1 = LEConv(n[0], n[1])
            self.conv2 = LEConv(n[1], n[2])
            self.conv3 = LEConv(n[2], n[3])
            self.conv4 = LEConv(n[3], n[4])
            self.conv5 = LEConv(n[4], n[5])
            self.conv6 = LEConv(n[5], n[6])
            self.conv7 = LEConv(n[6], n[7])
        elif self.type == 'PointNetConv':
            self.conv1 = PointNetConv(local_nn=Linear(n[0]+3, n[1]), global_nn=Linear(n[1], n[1]))
            self.conv2 = PointNetConv(local_nn=Linear(n[1]+3, n[2]), global_nn=Linear(n[2], n[2]))
            self.conv3 = PointNetConv(local_nn=Linear(n[2]+3, n[3]), global_nn=Linear(n[3], n[3]))
            self.conv4 = PointNetConv(local_nn=Linear(n[3]+3, n[4]), global_nn=Linear(n[4], n[4]))
            self.conv5 = PointNetConv(local_nn=Linear(n[4]+3, n[5]), global_nn=Linear(n[5], n[5]))
            self.conv6 = PointNetConv(local_nn=Linear(n[5]+3, n[6]), global_nn=Linear(n[6], n[6]))
            self.conv7 = PointNetConv(local_nn=Linear(n[6]+3, n[7]), global_nn=Linear(n[7], n[7]))


        self.norm1 = BatchNorm(in_channels=n[1])
        self.norm2 = BatchNorm(in_channels=n[2])

        self.norm3 = BatchNorm(in_channels=n[3])
        self.norm4 = BatchNorm(in_channels=n[4])

        self.norm5 = BatchNorm(in_channels=n[5])
        # self.pool5 = MaxPooling(self.pooling_size, transform=Cartesian(norm=True, cat=False), img_shape=self.input_shape[:2])

        self.norm6 = BatchNorm(in_channels=n[6])
        self.norm7 = BatchNorm(in_channels=n[7])

        # grid_div = 4  # =1: global_max_pool_x, >1: grid_max_pool_x
        num_grids = grid_div*grid_div
        pooling_dm_dims = torch.div(self.input_shape[:2], grid_div)
        self.pool7 = MaxPoolingX(pooling_dm_dims, size=num_grids, img_shape=self.input_shape[:2])
        self.fc = Linear(pooling_outputs * num_grids, out_features=num_outputs, bias=bias)

    def convs(self, layer, data):
        if self.type == 'SplineConv':
            return layer(data.x, data.edge_index, data.edge_attr)
        elif self.type == 'GCNConv':
            return layer(data.x, data.edge_index)
        elif self.type == 'LEConv':
            # return layer(data.x, data.edge_index, data.edge_attr)
            # should not use tensor edge_attr; should use a scalar
            raise NotImplementedError
        elif self.type == 'PointNetConv':
            return layer(data.x, data.pos, data.edge_index)



    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:

        data.x = self.norm1(self.convs(self.conv1, data))
        data.x = self.act(data.x)
        data.x = self.norm2(self.convs(self.conv2, data))
        data.x = self.act(data.x)

        x_sc = data.x.clone()
        data.x = self.norm3(self.convs(self.conv3, data))
        data.x = self.act(data.x)
        data.x = self.norm4(self.convs(self.conv4, data))
        data.x = self.act(data.x)
        data.x = data.x + x_sc

        data.x = self.norm5(self.convs(self.conv5, data))
        data.x = self.act(data.x)
        # data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = self.norm6(self.convs(self.conv6, data))
        data.x = self.act(data.x)
        data.x = self.norm7(self.convs(self.conv7, data))
        data.x = self.act(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features) # x.shape = [batch_size, num_grids*num_last_hidden_features]
        output = self.fc(x)
        return output
