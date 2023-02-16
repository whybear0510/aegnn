import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x, voxel_grid
from typing import List, Optional, Tuple, Union
from torch import Tensor
from .grid import fixed_voxel_grid


class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size: Tensor, size: int, img_shape: Tensor):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.full_shape = img_shape
        self.size = size


    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        # for a single sample
        if batch is None:
            batch = torch.zeros(pos.size(0), device=pos.device, dtype=torch.long)
        cluster = fixed_voxel_grid(pos, full_shape=self.full_shape, size=self.voxel_size, batch=batch)
        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"
