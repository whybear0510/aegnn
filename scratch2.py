import torch
import torch_geometric
from torch_geometric.data import Data
import pytorch_lightning as pl
import numpy as np
import aegnn
import os
from typing import Callable, List, Optional, Union
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian
from aegnn.models.layer import MaxPooling, MaxPoolingX
import glob
import os
from tqdm import tqdm

# torch.cuda.set_device(1)
device  = torch.device('cpu')

poses = []
training_files = glob.glob(r'/space/yyang22/datasets/data/storage/ncars/processed/training/*')

for i,tr_file in enumerate(tqdm(training_files)):
    data = torch.load(tr_file).to(device)
    poses.append(data.pos)


pos = torch.cat(poses, 0)
max_pos,_ = torch.max(pos,0)
min_pos,_ = torch.min(pos,0)
print(f'max_pos = \n{max_pos}')
print(f'min_pos = \n{min_pos}')
