import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import numpy as np
import os
import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import FixedPoints
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union
import aegnn
from aegnn.asyncronous import make_linear_asynchronous as async_lin

w = torch.tensor([[1,1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3,3],[4,4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5,5]], dtype=torch.float)
module = Linear(9,5, bias=False)
module.weight = torch.nn.Parameter(w)
x = torch.tensor([[0,1,2,3,4,5,6,7,8]], dtype=torch.float)
x_new = torch.tensor([[0,1,2,-3,-4,-5,-6,-7,-8]], dtype=torch.float)

y_sync = module.forward(x_new)
y_sync_init = module.forward(x)
print(y_sync_init)
print(y_sync)


module = async_lin(module)
y = module.forward(x)
print(y)
y_new = module.forward(x_new)
print(y_new)
print('')