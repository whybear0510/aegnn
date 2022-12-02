import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch.nn.functional import relu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX


class GraphRes(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False, act: str = 'elu'):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

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

        self.conv1 = GCNConv(n[0], n[1])
        self.norm1 = BatchNorm(in_channels=n[1])
        self.conv2 = GCNConv(n[1], n[2])
        self.norm2 = BatchNorm(in_channels=n[2])

        self.conv3 = GCNConv(n[2], n[3])
        self.norm3 = BatchNorm(in_channels=n[3])
        self.conv4 = GCNConv(n[3], n[4])
        self.norm4 = BatchNorm(in_channels=n[4])

        self.conv5 = GCNConv(n[4], n[5])
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.conv6 = GCNConv(n[5], n[6])
        self.norm6 = BatchNorm(in_channels=n[6])
        self.conv7 = GCNConv(n[6], n[7])
        self.norm7 = BatchNorm(in_channels=n[7])

        grid_div = 4  # =1: global_max_pool_x, >1: grid_max_pool_x
        num_grids = grid_div*grid_div
        pooling_dm_dims = torch.div(input_shape[:2], grid_div, rounding_mode='floor')
        self.pool7 = MaxPoolingX(pooling_dm_dims, size=num_grids)
        self.fc = Linear(pooling_outputs * num_grids, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:

        data.x = self.norm1(self.conv1(data.x, data.edge_index))
        # debug = data.x.detach().cpu().numpy()
        data.x = self.act(data.x)
        data.x = self.norm2(self.conv2(data.x, data.edge_index))
        data.x = self.act(data.x)

        x_sc = data.x.clone()
        data.x = self.norm3(self.conv3(data.x, data.edge_index))
        data.x = self.act(data.x)
        data.x = self.norm4(self.conv4(data.x, data.edge_index))
        data.x = self.act(data.x)
        data.x = data.x + x_sc

        data.x = self.norm5(self.conv5(data.x, data.edge_index))
        data.x = self.act(data.x)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = self.norm6(self.conv6(data.x, data.edge_index))
        data.x = self.act(data.x)
        data.x = self.norm7(self.conv7(data.x, data.edge_index))
        data.x = self.act(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features) # x.shape = [batch_size, num_grids*num_last_hidden_features]
        output = self.fc(x)
        return output
