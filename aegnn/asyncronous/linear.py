import numpy as np
import torch
import torch_geometric

from torch.nn import Linear
from .base.base import make_asynchronous, add_async_graph


def __graph_initialization(module: Linear, x: torch.Tensor) -> torch.Tensor:
    bias = getattr(module, 'bias', None)
    if bias is not None:
        y = x @ module.weight.T + bias
    else:
        y = x @ module.weight.T

    module.asy_graph = torch_geometric.data.Data(x=x.clone(), y=y)

    # If required, compute the flops of the asynchronous update operation.
    # flops computation from https://github.com/sovrasov/flops-counter.pytorch/
    if module.asy_flops_log is not None:
        flops = int(np.prod(x.size()) * y.size()[-1])
        module.asy_flops_log.append(flops)
    return module.asy_graph.y


def __graph_processing(module: Linear, x: torch.Tensor) -> torch.Tensor:
    diff_idx = (torch.nonzero(x - module.asy_graph.x).T)[1,:]
    if diff_idx.numel() > 0:
        x_diff = x[:, diff_idx] - module.asy_graph.x[:, diff_idx]
        partial_w = module.weight[:, diff_idx]
        y_diff = x_diff @ partial_w.T

        # Update the graph with the new values (only there where it has changed).
        module.asy_graph.x[:, diff_idx] = x[:, diff_idx]
        module.asy_graph.y += y_diff

    # If required, compute the flops of the asynchronous update operation.
    if module.asy_flops_log is not None:
        # flops = int(diff_idx.numel())
        # flops += y_diff.numel()  # graph update
        # module.asy_flops_log.append(flops)
        raise NotImplementedError('FLOPS for asy_linear waits to be implemented.')
    return module.asy_graph.y


def __check_support(module: Linear):
    return True


def make_linear_asynchronous(module: Linear, log_flops: bool = False, log_runtime: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for linear layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a linear layer can be converted by, for example:

    ```
    module = Linear(4, 2)
    module = make_linear_asynchronous(module)
    ```

    :param module: linear module to transform.
    :param log_flops: log flops of asynchronous update.
    :param log_runtime: log runtime of asynchronous update.
    """
    assert __check_support(module)
    module = add_async_graph(module, r=None, log_flops=log_flops, log_runtime=log_runtime)

    module.sync_graph = None #TODO: for debug
    return make_asynchronous(module, __graph_initialization, __graph_processing)
