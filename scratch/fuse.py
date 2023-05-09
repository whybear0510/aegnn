import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import functools
import glob
import logging
import numpy as np
import os
import torch
import importlib as imp

from tqdm import tqdm
tprint = tqdm.write

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import FixedPoints
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union

from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear

# from torch_geometric.nn.conv import PointNetConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairOptTensor,
    PairTensor
)
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.norm import BatchNorm
import pytorch_lightning as pl
import pandas as pd

import aegnn
# from aegnn.models.networks.my_fuse import MyConvBNReLU


pl.seed_everything(12345)
device = torch.device('cuda')

torch.set_printoptions(precision=6)

# path = "/users/yyang22/thesis/aegnn_project/aegnn_results/training_results/checkpoints/ncars/recognition/20230328183028/epoch=0-step=202.pt" #fuse
path = '/users/yyang22/thesis/aegnn_project/aegnn_results/training_results/checkpoints/ncars/recognition/20230330210943/epoch=36-step=7510.pt' # fuse, quant test
model = torch.load(path).to(device)
model.eval()
dm = aegnn.datasets.NCars(batch_size=1, shuffle=False)
dm.setup()
# print(model.model.fuse1.local_nn.weight)
data_loader = dm.val_dataloader(num_workers=1).__iter__()

assert model.model.fuse1.local_nn.bias is None

if isinstance(model, pl.LightningModule):
    nn_model = model._modules['model']
    nn_layers = nn_model._modules
elif isinstance(model, torch.nn.Module):
    nn_layers = model._modules
else:
    raise TypeError(f'The type of model is {type(model)}, not a `torch.nn.Module` or a `pl.LightningModule`')

from copy import deepcopy
unfused_model = deepcopy(model.model)
unfused_model = unfused_model.to(model.device)
unfused_model.eval()

for key, nn in nn_layers.items():
    if isinstance(nn, aegnn.models.networks.my_fuse.MyConvBNReLU):
        # nn_layers[key].module.running_mean = torch.zeros_like(nn_layers[key].module.running_mean)
        # nn_layers[key].module.running_var = torch.ones_like(nn_layers[key].module.running_var)
        # nn_layers[key].module.bias = torch.nn.Parameter(torch.zeros_like(nn_layers[key].module.bias))
        # nn_layers[key].module.weight = torch.nn.Parameter(torch.ones_like(nn_layers[key].module.weight))
        # nn_layers[key].module.eps = 1e-16
        nn_layers[key].to_fused()
        pass
fused_model = model
fused_model.eval()

num_test_samples = 4
# num_test_samples = 2400
with torch.no_grad():
    for i, sample in enumerate(tqdm(data_loader, position=1, desc='Samples', total=num_test_samples)):
        torch.cuda.empty_cache()
        if i==num_test_samples: break
        # tprint(f"\nSample {i}, file_id {sample.file_id}:")

        sample = sample.to(model.device)
        tot_nodes = sample.num_nodes


        unfused_test_sample = sample.clone().detach()
        output_unfused = unfused_model.forward(unfused_test_sample)
        y_unfused = torch.argmax(output_unfused, dim=-1)
        # tprint(f'unfused output = {output_unfused}')
        # tprint(f'{unfused_model.fuse1.local_nn.weight}')

        fused_test_sample = sample.clone().detach()
        output_fused = fused_model.forward(fused_test_sample)
        y_fused = torch.argmax(output_fused, dim=-1)
        # tprint(f'  fused output = {output_fused}')
        # tprint(f'{fused_model.model.fuse1.local_nn.weight}')
        # tprint(fused_model.model.fuse1.fused)

        diff = torch.allclose(y_unfused, y_fused)
        if diff is not True:
            print(i)
            print(f'unfused output = {output_unfused}')
            print(f'  fused output = {output_fused}')
        # print(diff)

