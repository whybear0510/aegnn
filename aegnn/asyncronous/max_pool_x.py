import torch
import torch_geometric

from aegnn.models.layer import MaxPoolingX
from .base.base import make_asynchronous, add_async_graph
from torch_geometric.data import Data
from typing import List, Optional, Tuple, Union
from torch import Tensor
from ..models.layer.grid import fixed_voxel_grid
from .base.utils import graph_changed_nodes, graph_new_nodes


def __graph_initialization(module: MaxPoolingX, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:

    y, cluster = module.sync_forward(x, pos, batch)

    # List all nodes idx that belong to certain grid
    idx_group = [torch.tensor([], device=x.device, dtype=torch.long) for i in range(module.size)]
    for i, c in enumerate(cluster):
        idx_group[c] = torch.cat([idx_group[c], torch.tensor([i], device=x.device, dtype=torch.long)])

    module.asy_graph = torch_geometric.data.Data(x=x.clone(), pos=pos, y=y, idx_group=idx_group)

    return module.asy_graph.y, cluster


def __graph_processing(module: MaxPoolingX, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Update pos with new added pos
    module.asy_graph.pos = pos_all = torch.cat([module.asy_graph.pos, pos], dim=0)

    # Identify the new added idx and changed idx
    _, idx_new = graph_new_nodes(module, x=x)
    _, idx_diff = graph_changed_nodes(module, x=x)
    idx_changed = torch.cat([idx_diff, idx_new])

    # Get new nodes' cluster, add them into idx_group
    cluster_new = fixed_voxel_grid(pos, full_shape=module.full_shape, size=module.voxel_size, batch=None)
    module.asy_graph.idx_group[cluster_new] = torch.cat([module.asy_graph.idx_group[cluster_new], idx_new])

    # Find the changed nodes' location in the grids, record the grids
    pos_changed = pos_all[idx_changed]
    cluster_changed = fixed_voxel_grid(pos_changed, full_shape=module.full_shape, size=module.voxel_size, batch=None)
    grid_changed = torch.unique(cluster_changed)

    # For each grid that contains changed nodes, get nodes' idx, find their x according to x, do the max(), then update y
    for g in grid_changed:
        idx_selected = module.asy_graph.idx_group[g]
        max, _ = torch.max(x[idx_selected,:], dim=0)
        module.asy_graph.y[g, :] = max

    # # Simplified method. Note: it is NOT a math restrict method
    # for g in grid_changed:
    #     former_max = module.asy_graph.y[g, :]
    #     changed_max,_ = torch.max(x[idx_changed, :], dim=0)
    #     new_max = torch.maximum(former_max, changed_max)
    #     module.asy_graph.y[g, :] = new_max

    # Update x
    module.asy_graph.x = x

    return module.asy_graph.y, cluster_changed


def __check_support(module: MaxPoolingX):
    return True


def make_max_pool_x_asynchronous(module: MaxPoolingX, log_flops: bool = False, log_runtime: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for MaxPoolingX layers.
    When nodes added/changed by former layers, according to their location in grids, compute feature-wise max pooling (only at affected grids):

    ```
    module = MaxPoolingX(voxel_size, size, img_shape)
    module = make_max_pool_x_asynchronous(module)
    ```

    :param module: MaxPoolingX module to transform.
    :param log_flops: log flops of asynchronous update.
    :param log_runtime: log runtime of asynchronous update.
    """
    assert __check_support(module)
    module = add_async_graph(module, r=None, log_flops=log_flops, log_runtime=log_runtime)
    module.sync_forward = module.forward

    return make_asynchronous(module, __graph_initialization, __graph_processing)
