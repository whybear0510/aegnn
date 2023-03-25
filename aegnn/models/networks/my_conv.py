from typing import Callable, Optional, Union

import torch
from torch import Tensor

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

class MyConv(MessagePassing):
    def __init__(self, local_nn: Optional[Callable] = None, global_nn: Optional[Callable] = None, add_self_loops: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        # super().__init__(local_nn, global_nn, add_self_loops, **kwargs)
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        reset(self.local_nn)
        reset(self.global_nn)


    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:
        # comment due to debug
        # msg = pos_j - pos_i
        # if x_j is not None:
        #     msg = torch.cat([x_j, msg], dim=1)
        # if self.local_nn is not None:
        #     msg = self.local_nn(msg)

        msg = torch.cat([x_j, torch.zeros_like(pos_j, dtype=x_j.dtype, device=x_j.device)], dim=1)
        msg = self.local_nn(msg) #TODO: just for debug
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')