import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch.nn.functional import relu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.conv import GCNConv, LEConv, PointNetConv, SAGEConv, GraphConv, GINConv, GINEConv, ARMAConv, SGConv, MFConv, NNConv, EdgeConv, ClusterGCNConv, GENConv, FiLMConv, PDNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian, Distance

from aegnn.models.layer import MaxPooling, MaxPoolingX

from .my_conv import MyConv


class GraphRes(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False, act: str = 'elu', grid_div: int = 8, conv_type: str = 'spline'):
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
            pooling_outputs = n[-1]
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")


        self.conv_type = conv_type

        if self.conv_type == 'spline':
            self.conv0 = SplineConv(1, 16, dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        elif self.conv_type == 'gcn':
            self.conv0 = GCNConv(1, 16, normalize=False, bias=False)
            # self.conv1 = GCNConv(n[0], n[1], normalize=False, bias=False)
            # self.conv2 = GCNConv(n[1], n[2], normalize=False, bias=False)
            # self.conv3 = GCNConv(n[2], n[3], normalize=False, bias=False)
            # self.conv4 = GCNConv(n[3], n[4], normalize=False, bias=False)
            self.conv5 = GCNConv(n[4], n[5], normalize=False, bias=False)
            self.conv6 = GCNConv(n[5], n[6], normalize=False, bias=False)
            self.conv7 = GCNConv(n[6], n[7], normalize=False, bias=False)
        elif self.conv_type == 'le':
            self.edge_weight_func = Distance(cat = True)
            self.conv1 = LEConv(n[0], n[1], bias=False)
            self.conv2 = LEConv(n[1], n[2], bias=False)
            self.conv3 = LEConv(n[2], n[3], bias=False)
            self.conv4 = LEConv(n[3], n[4], bias=False)
            self.conv5 = LEConv(n[4], n[5], bias=False)
            self.conv6 = LEConv(n[5], n[6], bias=False)
            self.conv7 = LEConv(n[6], n[7], bias=False)
        elif self.conv_type == 'pointnet':
            self.conv1 = PointNetConv(local_nn=Linear(n[0]+3, n[1]), global_nn=Linear(n[1], n[1]))
            self.conv2 = PointNetConv(local_nn=Linear(n[1]+3, n[2]), global_nn=Linear(n[2], n[2]))
            self.conv3 = PointNetConv(local_nn=Linear(n[2]+3, n[3]), global_nn=Linear(n[3], n[3]))
            self.conv4 = PointNetConv(local_nn=Linear(n[3]+3, n[4]), global_nn=Linear(n[4], n[4]))
            self.conv5 = PointNetConv(local_nn=Linear(n[4]+3, n[5]), global_nn=Linear(n[5], n[5]))
            self.conv6 = PointNetConv(local_nn=Linear(n[5]+3, n[6]), global_nn=Linear(n[6], n[6]))
            self.conv7 = PointNetConv(local_nn=Linear(n[6]+3, n[7]), global_nn=Linear(n[7], n[7]))
        elif self.conv_type == 'pointnet_single':
            self.conv0 = PointNetConv(local_nn=Linear(1+3, 16, bias=False), global_nn=None, add_self_loops=False)
            # self.conv1 = PointNetConv(local_nn=Linear(n[0]+3, n[1], bias=False), global_nn=None)
            # self.conv2 = PointNetConv(local_nn=Linear(n[1]+3, n[2], bias=False), global_nn=None)
            # self.conv3 = PointNetConv(local_nn=Linear(n[2]+3, n[3], bias=False), global_nn=None)
            # self.conv4 = PointNetConv(local_nn=Linear(n[3]+3, n[4], bias=False), global_nn=None)
            self.conv5 = PointNetConv(local_nn=Linear(n[4]+3, n[5], bias=False), global_nn=None, add_self_loops=False)
            self.conv6 = PointNetConv(local_nn=Linear(n[5]+3, n[6], bias=False), global_nn=None, add_self_loops=False)
            self.conv7 = PointNetConv(local_nn=Linear(n[6]+3, n[7], bias=False), global_nn=None, add_self_loops=False)
            # self.conv8 = PointNetConv(local_nn=Linear(n[7]+3, n[8], bias=False), global_nn=None)
        elif self.conv_type == 'my':
            self.conv0 = MyConv(local_nn=Linear(1+3, 16, bias=False), global_nn=None)
            self.conv5 = MyConv(local_nn=Linear(n[4]+3, n[5], bias=False), global_nn=None)
            self.conv6 = MyConv(local_nn=Linear(n[5]+3, n[6], bias=False), global_nn=None)
            self.conv7 = MyConv(local_nn=Linear(n[6]+3, n[7], bias=False), global_nn=None)
        elif self.conv_type == 'sage':
            self.conv1 = SAGEConv(n[0], n[1])
            self.conv2 = SAGEConv(n[1], n[2])
            self.conv3 = SAGEConv(n[2], n[3])
            self.conv4 = SAGEConv(n[3], n[4])
            self.conv5 = SAGEConv(n[4], n[5])
            self.conv6 = SAGEConv(n[5], n[6])
            self.conv7 = SAGEConv(n[6], n[7])
        elif self.conv_type == 'graph':
            self.edge_weight_func = Distance(cat = True)
            self.conv1 = GraphConv(n[0], n[1])
            self.conv2 = GraphConv(n[1], n[2])
            self.conv3 = GraphConv(n[2], n[3])
            self.conv4 = GraphConv(n[3], n[4])
            self.conv5 = GraphConv(n[4], n[5])
            self.conv6 = GraphConv(n[5], n[6])
            self.conv7 = GraphConv(n[6], n[7])
        elif self.conv_type == 'gin':
            self.conv1 = GINConv(nn=Linear(n[0], n[1]))
            self.conv2 = GINConv(nn=Linear(n[1], n[2]))
            self.conv3 = GINConv(nn=Linear(n[2], n[3]))
            self.conv4 = GINConv(nn=Linear(n[3], n[4]))
            self.conv5 = GINConv(nn=Linear(n[4], n[5]))
            self.conv6 = GINConv(nn=Linear(n[5], n[6]))
            self.conv7 = GINConv(nn=Linear(n[6], n[7]))
        elif self.conv_type == 'gine':
            # self.edge_weight_func = Distance(cat = False)
            # self.conv1 = GINEConv(nn=Linear(n[0], n[1]), edge_dim=n[0])
            # self.conv2 = GINEConv(nn=Linear(n[1], n[2]), edge_dim=n[1])
            # self.conv3 = GINEConv(nn=Linear(n[2], n[3]), edge_dim=n[2])
            # self.conv4 = GINEConv(nn=Linear(n[3], n[4]), edge_dim=n[3])
            # self.conv5 = GINEConv(nn=Linear(n[4], n[5]), edge_dim=n[4])
            # self.conv6 = GINEConv(nn=Linear(n[5], n[6]), edge_dim=n[5])
            # self.conv7 = GINEConv(nn=Linear(n[6], n[7]), edge_dim=n[6])
            raise NotImplementedError("GINEConv contains some bug(?)")
        elif self.conv_type == 'arma':
            self.conv1 = ARMAConv(n[0], n[1], num_stacks=2, num_layers=2)
            self.conv2 = ARMAConv(n[1], n[2], num_stacks=2, num_layers=2)
            self.conv3 = ARMAConv(n[2], n[3], num_stacks=2, num_layers=2)
            self.conv4 = ARMAConv(n[3], n[4], num_stacks=2, num_layers=2)
            self.conv5 = ARMAConv(n[4], n[5], num_stacks=2, num_layers=2)
            self.conv6 = ARMAConv(n[5], n[6], num_stacks=2, num_layers=2)
            self.conv7 = ARMAConv(n[6], n[7], num_stacks=2, num_layers=2)
        elif self.conv_type == 'sg' or self.conv_type == 'sg_weighted':
            self.edge_weight_func = Distance(cat = True)
            self.conv1 = SGConv(n[0], n[1], K=1)
            self.conv2 = SGConv(n[1], n[2], K=1)
            self.conv3 = SGConv(n[2], n[3], K=1)
            self.conv4 = SGConv(n[3], n[4], K=1)
            self.conv5 = SGConv(n[4], n[5], K=1)
            self.conv6 = SGConv(n[5], n[6], K=1)
            self.conv7 = SGConv(n[6], n[7], K=1)
        elif self.conv_type == 'mf':
            self.conv1 = MFConv(n[0], n[1])
            self.conv2 = MFConv(n[1], n[2])
            self.conv3 = MFConv(n[2], n[3])
            self.conv4 = MFConv(n[3], n[4])
            self.conv5 = MFConv(n[4], n[5])
            self.conv6 = MFConv(n[5], n[6])
            self.conv7 = MFConv(n[6], n[7])
        elif self.conv_type == 'nn':
            self.conv1 = NNConv(n[0], n[1], bias=bias, root_weight=root_weight, nn=Linear(3,n[0]*n[1]))
            self.conv2 = NNConv(n[1], n[2], bias=bias, root_weight=root_weight, nn=Linear(3,n[1]*n[2]))
            self.conv3 = NNConv(n[2], n[3], bias=bias, root_weight=root_weight, nn=Linear(3,n[2]*n[3]))
            self.conv4 = NNConv(n[3], n[4], bias=bias, root_weight=root_weight, nn=Linear(3,n[3]*n[4]))
            self.conv5 = NNConv(n[4], n[5], bias=bias, root_weight=root_weight, nn=Linear(3,n[4]*n[5]))
            self.conv6 = NNConv(n[5], n[6], bias=bias, root_weight=root_weight, nn=Linear(3,n[5]*n[6]))
            self.conv7 = NNConv(n[6], n[7], bias=bias, root_weight=root_weight, nn=Linear(3,n[6]*n[7]))
        elif self.conv_type == 'edge':
            self.conv1 = EdgeConv(nn=Linear(2*n[0], n[1]))
            self.conv2 = EdgeConv(nn=Linear(2*n[1], n[2]))
            self.conv3 = EdgeConv(nn=Linear(2*n[2], n[3]))
            self.conv4 = EdgeConv(nn=Linear(2*n[3], n[4]))
            self.conv5 = EdgeConv(nn=Linear(2*n[4], n[5]))
            self.conv6 = EdgeConv(nn=Linear(2*n[5], n[6]))
            self.conv7 = EdgeConv(nn=Linear(2*n[6], n[7]))
        elif self.conv_type == 'cluster':
            self.conv1 = ClusterGCNConv(n[0], n[1])
            self.conv2 = ClusterGCNConv(n[1], n[2])
            self.conv3 = ClusterGCNConv(n[2], n[3])
            self.conv4 = ClusterGCNConv(n[3], n[4])
            self.conv5 = ClusterGCNConv(n[4], n[5])
            self.conv6 = ClusterGCNConv(n[5], n[6])
            self.conv7 = ClusterGCNConv(n[6], n[7])
        elif self.conv_type == 'film':
            self.conv1 = FiLMConv(n[0], n[1])
            self.conv2 = FiLMConv(n[1], n[2])
            self.conv3 = FiLMConv(n[2], n[3])
            self.conv4 = FiLMConv(n[3], n[4])
            self.conv5 = FiLMConv(n[4], n[5])
            self.conv6 = FiLMConv(n[5], n[6])
            self.conv7 = FiLMConv(n[6], n[7])
        elif self.conv_type == 'pdn':
            self.conv1 = PDNConv(n[0], n[1], edge_dim=3, hidden_channels=4, normalize=False)
            self.conv2 = PDNConv(n[1], n[2], edge_dim=3, hidden_channels=4, normalize=False)
            self.conv3 = PDNConv(n[2], n[3], edge_dim=3, hidden_channels=4, normalize=False)
            self.conv4 = PDNConv(n[3], n[4], edge_dim=3, hidden_channels=4, normalize=False)
            self.conv5 = PDNConv(n[4], n[5], edge_dim=3, hidden_channels=4, normalize=False)
            self.conv6 = PDNConv(n[5], n[6], edge_dim=3, hidden_channels=4, normalize=False)
            self.conv7 = PDNConv(n[6], n[7], edge_dim=3, hidden_channels=4, normalize=False)
        elif self.conv_type == 'gcn_weighted':
            self.edge_weight_func = Distance(cat = True)
            self.conv1 = GCNConv(n[0], n[1], normalize=False, bias=False)
            self.conv2 = GCNConv(n[1], n[2], normalize=False, bias=False)
            self.conv3 = GCNConv(n[2], n[3], normalize=False, bias=False)
            self.conv4 = GCNConv(n[3], n[4], normalize=False, bias=False)
            self.conv5 = GCNConv(n[4], n[5], normalize=False, bias=False)
            self.conv6 = GCNConv(n[5], n[6], normalize=False, bias=False)
            self.conv7 = GCNConv(n[6], n[7], normalize=False, bias=False)
        else:
            raise ValueError(f"Unkown convolution type: {self.conv_type}")


        # self.norm1 = BatchNorm(in_channels=n[1])
        # self.norm2 = BatchNorm(in_channels=n[2])

        # self.norm3 = BatchNorm(in_channels=n[3])
        # self.norm4 = BatchNorm(in_channels=n[4])

        self.norm0 = BatchNorm(in_channels=16)
        self.norm5 = BatchNorm(in_channels=n[5])
        # self.pool5 = MaxPooling(self.pooling_size, transform=Cartesian(norm=True, cat=False), img_shape=self.input_shape[:2])

        self.norm6 = BatchNorm(in_channels=n[6])
        self.norm7 = BatchNorm(in_channels=n[7])
        self.norm8 = BatchNorm(in_channels=n[8])

        # grid_div = 4  # =1: global_max_pool_x, >1: grid_max_pool_x
        num_grids = grid_div*grid_div
        pooling_dm_dims = torch.div(self.input_shape[:2], grid_div)
        self.pool7 = MaxPoolingX(pooling_dm_dims, size=num_grids, img_shape=self.input_shape[:2])
        self.fc = Linear(pooling_outputs * num_grids, out_features=num_outputs, bias=False)
        # self.hidden = 128
        # self.fc1 = Linear(pooling_outputs * num_grids, out_features=self.hidden, bias=False)
        # self.fc2 = Linear(self.hidden, out_features=num_outputs, bias=False)

    def convs(self, layer, data):
        if self.conv_type == 'spline' or self.conv_type == 'gine' or self.conv_type == 'nn' or self.conv_type == 'pdn':
            return layer(data.x, data.edge_index, data.edge_attr)
        elif self.conv_type == 'gcn' \
          or self.conv_type == 'sage' \
          or self.conv_type == 'gin' \
          or self.conv_type == 'arma' \
          or self.conv_type == 'sg' \
          or self.conv_type == 'mf' \
          or self.conv_type == 'edge' \
          or self.conv_type == 'cluster' \
          or self.conv_type == 'film':
            return layer(data.x, data.edge_index)
        elif self.conv_type == 'le' or self.conv_type == 'graph' or self.conv_type == 'sg_weighted' or self.conv_type == 'gcn_weighted':
            return layer(data.x, data.edge_index, data.edge_weight)
        elif self.conv_type == 'pointnet' or self.conv_type == 'pointnet_single' or self.conv_type == 'my':
            return layer(x=data.x, pos=data.pos, edge_index=data.edge_index)
        else:
            raise ValueError(f"Unkown convolution type: {self.conv_type}")



    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        assert self.conv_type == 'le' \
            or self.conv_type == 'spline' \
            or self.conv_type == 'gcn' \
            or self.conv_type == 'pointnet' \
            or self.conv_type == 'sage' \
            or self.conv_type == 'graph' \
            or self.conv_type == 'gin' \
            or self.conv_type == 'arma' \
            or self.conv_type == 'sg' \
            or self.conv_type == 'sg_weighted' \
            or self.conv_type == 'mf' \
            or self.conv_type == 'edge' \
            or self.conv_type == 'cluster' \
            or self.conv_type == 'film' \
            or self.conv_type == 'pdn' \
            or self.conv_type == 'gcn_weighted' \
            or self.conv_type == 'pointnet_single' \
            or self.conv_type == 'my'
            # or self.conv_type == 'gen'
            # or self.conv_type == 'nn'
            # or self.conv_type == 'gine'


        if self.conv_type == 'le' or self.conv_type == 'graph' or self.conv_type == 'sg_weighted' or self.conv_type == 'gcn_weighted':
            data = self.edge_weight_func(data)
            data.edge_weight = data.edge_attr[:,-1]
            data.edge_attr = data.edge_attr[:, :-1]
        elif self.conv_type == 'gine':
            data = self.edge_weight_func(data)

        # data.x = self.convs(self.conv1, data)
        # data.x = self.norm1(data.x)
        # data.x = self.act(data.x)
        # data.x = self.norm2(self.convs(self.conv2, data))
        # data.x = self.act(data.x)

        data.x = self.convs(self.conv0, data)
        data.x = self.norm0(data.x)
        data.x = self.act(data.x)

        # x_sc = data.x.clone()
        # data.x = self.norm3(self.convs(self.conv3, data))
        # data.x = self.act(data.x)
        # data.x = self.norm4(self.convs(self.conv4, data))
        # data.x = self.act(data.x)
        # data.x = data.x + x_sc

        data.x = self.norm5(self.convs(self.conv5, data))
        data.x = self.act(data.x)
        # data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = self.norm6(self.convs(self.conv6, data))
        data.x = self.act(data.x)
        data.x = self.norm7(self.convs(self.conv7, data))
        data.x = self.act(data.x)
        data.x = data.x + x_sc

        # data.x = self.norm8(self.convs(self.conv8, data))
        # data.x = self.act(data.x)

        x,_ = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)

        x = x.view(-1, self.fc.in_features) # x.shape = [batch_size, num_grids*num_last_hidden_features]
        output = self.fc(x)

        # o1 = self.fc1(x)
        # o1_relu = self.act(o1)
        # output = self.fc2(o1_relu)
        return output
