import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch.nn.functional import relu
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import SplineConv, GCNConv, LEConv, PointNetConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian, Distance

from aegnn.models.layer import MaxPooling, MaxPoolingX

from .my_conv import MyConv
from .my_fuse import MyConvBNReLU, qLinear

from aegnn.utils import Qtype


class GraphRes(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False, act: str = 'relu', grid_div: int = 8, conv_type: str = 'fuse'):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # TODO: more elegant way to use pl "self.device"
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        # self.pooling_size = torch.tensor(pooling_size, device=self.device)
        self.input_shape = input_shape.to(self.device)

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

        self.fused = False
        self.quantized = False

        self.conv_type = conv_type

        if self.conv_type == 'spline':
            self.conv1 = SplineConv(1, 16, dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv2 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv3 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
            self.conv4 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=False, root_weight=False)
        elif self.conv_type == 'gcn':
            self.conv1 = GCNConv(1, 16, normalize=False, bias=False)
            self.conv2 = GCNConv(n[4], n[5], normalize=False, bias=False)
            self.conv3 = GCNConv(n[5], n[6], normalize=False, bias=False)
            self.conv4 = GCNConv(n[6], n[7], normalize=False, bias=False)
        elif self.conv_type == 'le':
            self.edge_weight_func = Distance(cat = True)
            self.conv1 = LEConv(1, 16, bias=False)
            self.conv2 = LEConv(n[4], n[5], bias=False)
            self.conv3 = LEConv(n[5], n[6], bias=False)
            self.conv4 = LEConv(n[6], n[7], bias=False)
        elif self.conv_type == 'pointnet':
            self.conv1 = PointNetConv(local_nn=Linear(1+3, 16), global_nn=Linear(16, 16))
            self.conv2 = PointNetConv(local_nn=Linear(n[4]+3, n[5]), global_nn=Linear(n[5], n[5]))
            self.conv3 = PointNetConv(local_nn=Linear(n[5]+3, n[6]), global_nn=Linear(n[6], n[6]))
            self.conv4 = PointNetConv(local_nn=Linear(n[6]+3, n[7]), global_nn=Linear(n[7], n[7]))
        elif self.conv_type == 'pointnet_single':
            self.conv1 = PointNetConv(local_nn=Linear(1+3, 16, bias=False), global_nn=None, add_self_loops=False)
            self.conv2 = PointNetConv(local_nn=Linear(n[4]+3, n[5], bias=False), global_nn=None, add_self_loops=False)
            self.conv3 = PointNetConv(local_nn=Linear(n[5]+3, n[6], bias=False), global_nn=None, add_self_loops=False)
            self.conv4 = PointNetConv(local_nn=Linear(n[6]+3, n[7], bias=False), global_nn=None, add_self_loops=False)
        elif self.conv_type == 'my':
            self.conv1 = MyConv(1, 16)
            self.conv2 = MyConv(n[4], n[5])
            self.conv3 = MyConv(n[5], n[6])
            self.conv4 = MyConv(n[6], n[7])


        if self.conv_type != 'fuse':
            self.norm1 = BatchNorm(in_channels=16)
            self.norm2 = BatchNorm(in_channels=n[5])
            self.norm3 = BatchNorm(in_channels=n[6])
            self.norm4 = BatchNorm(in_channels=n[7])
        elif self.conv_type == 'fuse':
            # print('Fuse mode: conv, bn, relu')
            self.fuse1 = MyConvBNReLU(1, 16)
            self.fuse2 = MyConvBNReLU(n[4], n[5])
            self.fuse3 = MyConvBNReLU(n[5], n[6])
            self.fuse4 = MyConvBNReLU(n[6], n[7])
        else:
            raise ValueError(f"Unkown convolution type: {self.conv_type}")

        # grid_div = 4  # =1: global_max_pool_x, >1: grid_max_pool_x
        num_grids = grid_div*grid_div
        pooling_dm_dims = torch.div(self.input_shape[:2], grid_div)
        self.pool = MaxPoolingX(pooling_dm_dims, size=num_grids, img_shape=self.input_shape[:2])
        # self.fc = Linear(pooling_outputs * num_grids, out_features=num_outputs, bias=False)
        self.fc = qLinear(pooling_outputs * num_grids, out_features=num_outputs, bias=False)
        # self.hidden = 128
        # self.fc1 = Linear(pooling_outputs * num_grids, out_features=self.hidden, bias=False)
        # self.fc2 = Linear(self.hidden, out_features=num_outputs, bias=False)


    def convs(self, layer, data):
        if self.conv_type == 'spline':
            return layer(data.x, data.edge_index, data.edge_attr)
        elif self.conv_type == 'gcn':
            return layer(data.x, data.edge_index)
        elif self.conv_type == 'le':
            return layer(data.x, data.edge_index, data.edge_weight)
        elif self.conv_type == 'pointnet' or self.conv_type == 'pointnet_single' or self.conv_type == 'my':
            return layer(x=data.x, pos=data.pos, edge_index=data.edge_index)
        else:
            raise ValueError(f"Unkown convolution type: {self.conv_type}")

    def to_fused(self):
        for module in self.children():
            if isinstance(module, MyConvBNReLU):
                module.to_fused()
        self.fused = True
        return self

    def quant(self,*,conv_f_dtype='uint8', conv_w_dtype='int8', fc_in_dtype='uint8', fc_w_dtype='int8', fc_out_dtype='int8'):
        for module in self.children():
            if isinstance(module, MyConvBNReLU):
                module.quant(f_dtype=conv_f_dtype, w_dtype=conv_w_dtype)
            elif isinstance(module, qLinear):
                module.quant(in_dtype=fc_in_dtype, w_dtype=fc_w_dtype, out_dtype=fc_out_dtype)
        self.quantized = True
        return self

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # assert data.x.device.type == data.pos.device.type == data.edge_index.device.type == self.device.type

        assert self.conv_type == 'le' \
            or self.conv_type == 'spline' \
            or self.conv_type == 'gcn' \
            or self.conv_type == 'pointnet' \
            or self.conv_type == 'pointnet_single' \
            or self.conv_type == 'my' \
            or self.conv_type == 'fuse'


        if self.conv_type == 'le' :
            data = self.edge_weight_func(data)
            data.edge_weight = data.edge_attr[:,-1]
            data.edge_attr = data.edge_attr[:, :-1]

        if self.conv_type != 'fuse':
            data.x = self.convs(self.conv1, data)
            data.x = self.norm1(data.x)
            data.x = self.act(data.x)

            data.x = self.norm2(self.convs(self.conv2, data))
            data.x = self.act(data.x)

            data.x = self.norm3(self.convs(self.conv3, data))
            data.x = self.act(data.x)

            data.x = self.norm4(self.convs(self.conv4, data))
            data.x = self.act(data.x)


        elif self.conv_type == 'fuse':
            if not self.quantized:
                data.x = self.fuse1(x=data.x, pos=data.pos[:,:2], edge_index=data.edge_index) # no timestamp
                data.x = self.fuse2(x=data.x, pos=data.pos[:,:2], edge_index=data.edge_index)
                data.x = self.fuse3(x=data.x, pos=data.pos[:,:2], edge_index=data.edge_index)
                data.x = self.fuse4(x=data.x, pos=data.pos[:,:2], edge_index=data.edge_index)
            else:
                data.x = MyConvBNReLU.quant_tensor(data.x, scale=self.fuse1.x_scale, dtype=self.fuse1.f_dtype)
                data.x = self.fuse1(x=data.x, pos=data.pos[:,:2], edge_index=data.edge_index)
                data.x = self.fuse2(x=data.x, pos=data.pos[:,:2], edge_index=data.edge_index)
                data.x = self.fuse3(x=data.x, pos=data.pos[:,:2], edge_index=data.edge_index)
                data.x = self.fuse4(x=data.x, pos=data.pos[:,:2], edge_index=data.edge_index)
                # data.x = MyConvBNReLU.dequant_tensor(data.x, scale=self.fuse4.y_scale)



        x,_ = self.pool(data.x, pos=data.pos[:, :2], batch=data.batch)

        x = x.view(-1, self.fc.in_features) # x.shape = [batch_size, num_grids*num_last_hidden_features]

        if not self.quantized:
            output = self.fc(x)
        else:
            q_output = self.fc(x)
            # dq_output = MyConvBNReLU.dequant_tensor(q_output, scale=self.fc.out_scale)
            # output = dq_output
            output = q_output


        # o1 = self.fc1(x)
        # o1_relu = self.act(o1)
        # output = self.fc2(o1_relu)
        return output

    def debug_logger(self):
        self.debug_y = {}
        self.debug_qy = {}
        self.debug_dqy = {}

        self.debug_fc = {}

        if self.conv_type == 'fuse':
            def log_y(name):
                def hook(module, input, output):
                    if not self.quantized:
                        self.debug_y[name] = output.detach()
                    else:
                        self.debug_qy[name] = output.detach()
                        self.debug_dqy[name] = MyConvBNReLU.dequant_tensor(output.detach(), scale=module.y_scale)
                return hook

            def log_io():
                def hook(module, input, output):
                    if not self.quantized:
                        self.debug_fc['in'] = input[0].detach()
                        self.debug_fc['out'] = output.detach()
                    else:
                        self.debug_fc['qin'] = input[0].detach()
                        self.debug_fc['dqin'] = MyConvBNReLU.dequant_tensor(input[0].detach(), scale=module.in_scale)
                        self.debug_fc['qout'] = output.detach()
                        self.debug_fc['dqout'] = MyConvBNReLU.dequant_tensor(output.detach(), scale=module.out_scale)
                return hook


            for name, module in self.named_children():
                if not name.startswith('params'):
                    if isinstance(module, MessagePassing):
                        module.register_forward_hook(log_y(name))
                    if isinstance(module, qLinear):
                        module.register_forward_hook(log_io())


