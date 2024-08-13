"""
    Architectural common routines for models in PyTorch.
"""

__all__ = ['DualPathSequential', 'Concurrent', 'SequentialConcurrent', 'ParametricSequential', 'ParametricConcurrent',
           'Hourglass', 'SesquialteralHourglass', 'MultiOutputSequential', 'ParallelConcurent',
           'DualPathParallelConcurent']

import torch
import torch.nn as nn
from typing import Callable


class DualPathSequential(nn.Sequential):
    """
    A sequential container for modules with dual inputs/outputs.
    Modules will be executed in the order they are added.

    Parameters
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first modules with single input/output.
    last_ordinals : int, default 0
        Number of the final modules with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a module.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal module.
    """
    def __init__(self,
                 return_two: bool = True,
                 first_ordinals: int = 0,
                 last_ordinals: int = 0,
                 dual_path_scheme: Callable = (lambda module, x1, x2: module(x1, x2)),
                 dual_path_scheme_ordinal: Callable = (lambda module, x1, x2: (module(x1), x2))):
        super(DualPathSequential, self).__init__()
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def forward(self, x1, x2=None):
        length = len(self._modules.values())
        for i, module in enumerate(self._modules.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(module, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(module, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class Concurrent(nn.Sequential):
    """
    A container for concatenation of modules on the base of the sequential container.

    Parameters
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    merge_type : str or None, default None
        Type of branch merging.
    """
    def __init__(self,
                 axis: int = 1,
                 stack: bool = False,
                 merge_type: str | None = None):
        super(Concurrent, self).__init__()
        assert (merge_type is None) or (merge_type in ["cat", "stack", "sum"])
        self.axis = axis
        if merge_type is not None:
            self.merge_type = merge_type
        else:
            self.merge_type = "stack" if stack else "cat"

    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        if self.merge_type == "stack":
            out = torch.stack(tuple(out), dim=self.axis)
        elif self.merge_type == "cat":
            out = torch.cat(tuple(out), dim=self.axis)
        elif self.merge_type == "sum":
            out = torch.stack(tuple(out), dim=self.axis).sum(self.axis)
        else:
            raise NotImplementedError()
        return out


class SequentialConcurrent(nn.Sequential):
    """
    A sequential container with concatenated outputs.
    Modules will be executed in the order they are added.

    Parameters
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    cat_input : bool, default True
        Whether to concatenate input tensor.
    """
    def __init__(self,
                 axis: int = 1,
                 stack: bool = False,
                 cat_input: bool = True):
        super(SequentialConcurrent, self).__init__()
        self.axis = axis
        self.stack = stack
        self.cat_input = cat_input

    def forward(self, x):
        out = [x] if self.cat_input else []
        for module in self._modules.values():
            x = module(x)
            out.append(x)
        if self.stack:
            out = torch.stack(tuple(out), dim=self.axis)
        else:
            out = torch.cat(tuple(out), dim=self.axis)
        return out


class ParametricSequential(nn.Sequential):
    """
    A sequential container for modules with parameters.
    Modules will be executed in the order they are added.
    """
    def __init__(self, *args):
        super(ParametricSequential, self).__init__(*args)

    def forward(self, x, **kwargs):
        for module in self._modules.values():
            x = module(x, **kwargs)
        return x


class ParametricConcurrent(nn.Sequential):
    """
    A container for concatenation of modules with parameters.

    Parameters
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis: int = 1):
        super(ParametricConcurrent, self).__init__()
        self.axis = axis

    def forward(self, x, **kwargs):
        out = []
        for module in self._modules.values():
            out.append(module(x, **kwargs))
        out = torch.cat(tuple(out), dim=self.axis)
        return out


class Hourglass(nn.Module):
    """
    An hourglass module.

    Parameters
    ----------
    down_seq : nn.Sequential
        Down modules as sequential.
    up_seq : nn.Sequential
        Up modules as sequential.
    skip_seq : nn.Sequential
        Skip connection modules as sequential.
    merge_type : str, default 'add'
        Type of concatenation of up and skip outputs.
    return_first_skip : bool, default False
        Whether return the first skip connection output. Used in ResAttNet.
    """
    def __init__(self,
                 down_seq: nn.Sequential,
                 up_seq: nn.Sequential,
                 skip_seq: nn.Sequential,
                 merge_type: str = "add",
                 return_first_skip: bool = False):
        super(Hourglass, self).__init__()
        self.depth = len(down_seq)
        assert (merge_type in ["cat", "add"])
        assert (len(up_seq) == self.depth)
        assert (len(skip_seq) in (self.depth, self.depth + 1))
        self.merge_type = merge_type
        self.return_first_skip = return_first_skip
        self.extra_skip = (len(skip_seq) == self.depth + 1)

        self.down_seq = down_seq
        self.up_seq = up_seq
        self.skip_seq = skip_seq

    def _merge(self, x, y):
        if y is not None:
            if self.merge_type == "cat":
                x = torch.cat((x, y), dim=1)
            elif self.merge_type == "add":
                x = x + y
        return x

    def forward(self, x, **kwargs):
        y = None
        down_outs = [x]
        for down_module in self.down_seq._modules.values():
            x = down_module(x)
            down_outs.append(x)
        for i in range(len(down_outs)):
            if i != 0:
                y = down_outs[self.depth - i]
                skip_module = self.skip_seq[self.depth - i]
                y = skip_module(y)
                x = self._merge(x, y)
            if i != len(down_outs) - 1:
                if (i == 0) and self.extra_skip:
                    skip_module = self.skip_seq[self.depth]
                    x = skip_module(x)
                up_module = self.up_seq[self.depth - 1 - i]
                x = up_module(x)
        if self.return_first_skip:
            return x, y
        else:
            return x


class SesquialteralHourglass(nn.Module):
    """
    A sesquialteral hourglass block.

    Parameters
    ----------
    down1_seq : nn.Sequential
        The first down modules as sequential.
    skip1_seq : nn.Sequential
        The first skip connection modules as sequential.
    up_seq : nn.Sequential
        Up modules as sequential.
    skip2_seq : nn.Sequential
        The second skip connection modules as sequential.
    down2_seq : nn.Sequential
        The second down modules as sequential.
    merge_type : str, default 'cat'
        Type of concatenation of up and skip outputs.
    """
    def __init__(self,
                 down1_seq: nn.Sequential,
                 skip1_seq: nn.Sequential,
                 up_seq: nn.Sequential,
                 skip2_seq: nn.Sequential,
                 down2_seq: nn.Sequential,
                 merge_type: str = "cat"):
        super(SesquialteralHourglass, self).__init__()
        assert (len(down1_seq) == len(up_seq))
        assert (len(down1_seq) == len(down2_seq))
        assert (len(skip1_seq) == len(skip2_seq))
        assert (len(down1_seq) == len(skip1_seq) - 1)
        assert (merge_type in ["cat", "add"])
        self.merge_type = merge_type
        self.depth = len(down1_seq)

        self.down1_seq = down1_seq
        self.skip1_seq = skip1_seq
        self.up_seq = up_seq
        self.skip2_seq = skip2_seq
        self.down2_seq = down2_seq

    def _merge(self, x, y):
        if y is not None:
            if self.merge_type == "cat":
                x = torch.cat((x, y), dim=1)
            elif self.merge_type == "add":
                x = x + y
        return x

    def forward(self, x, **kwargs):
        y = self.skip1_seq[0](x)
        skip1_outs = [y]
        for i in range(self.depth):
            x = self.down1_seq[i](x)
            y = self.skip1_seq[i + 1](x)
            skip1_outs.append(y)
        x = skip1_outs[self.depth]
        y = self.skip2_seq[0](x)
        skip2_outs = [y]
        for i in range(self.depth):
            x = self.up_seq[i](x)
            y = skip1_outs[self.depth - 1 - i]
            x = self._merge(x, y)
            y = self.skip2_seq[i + 1](x)
            skip2_outs.append(y)
        x = self.skip2_seq[self.depth](x)
        for i in range(self.depth):
            x = self.down2_seq[i](x)
            y = skip2_outs[self.depth - 1 - i]
            x = self._merge(x, y)
        return x


class MultiOutputSequential(nn.Sequential):
    """
    A sequential container with multiple outputs.
    Modules will be executed in the order they are added.

    Parameters
    ----------
    multi_output : bool, default True
        Whether to return multiple output.
    dual_output : bool, default False
        Whether to return dual output.
    return_last : bool, default True
        Whether to forcibly return last value.
    """
    def __init__(self,
                 multi_output: bool = True,
                 dual_output: bool = False,
                 return_last: bool = True):
        super(MultiOutputSequential, self).__init__()
        self.multi_output = multi_output
        self.dual_output = dual_output
        self.return_last = return_last

    def forward(self, x):
        outs = []
        for module in self._modules.values():
            x = module(x)
            if hasattr(module, "do_output") and module.do_output:
                outs.append(x)
            elif hasattr(module, "do_output2") and module.do_output2:
                assert isinstance(x, tuple)
                outs.extend(x[1])
                x = x[0]
        if self.multi_output:
            return [x] + outs if self.return_last else outs
        elif self.dual_output:
            return x, outs
        else:
            return x


class ParallelConcurent(nn.Sequential):
    """
    A sequential container with multiple inputs and single/multiple outputs.
    Modules will be executed in the order they are added.

    Parameters
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    merge_type : str, default 'list'
        Type of branch merging.
    """
    def __init__(self,
                 axis: int = 1,
                 merge_type: str = "list"):
        super(ParallelConcurent, self).__init__()
        assert (merge_type is None) or (merge_type in ["list", "cat", "stack", "sum"])
        self.axis = axis
        self.merge_type = merge_type

    def forward(self, x):
        out = []
        for module, xi in zip(self._modules.values(), x):
            out.append(module(xi))
        if self.merge_type == "list":
            pass
        elif self.merge_type == "stack":
            out = torch.stack(tuple(out), dim=self.axis)
        elif self.merge_type == "cat":
            out = torch.cat(tuple(out), dim=self.axis)
        elif self.merge_type == "sum":
            out = torch.stack(tuple(out), dim=self.axis).sum(self.axis)
        else:
            raise NotImplementedError()
        return out


class DualPathParallelConcurent(nn.Sequential):
    """
    A sequential container with multiple dual-path inputs and single/multiple outputs.
    Modules will be executed in the order they are added.

    Parameters
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    merge_type : str, default 'list'
        Type of branch merging.
    """
    def __init__(self,
                 axis: int = 1,
                 merge_type: str = "list"):
        super(DualPathParallelConcurent, self).__init__()
        assert (merge_type is None) or (merge_type in ["list", "cat", "stack", "sum"])
        self.axis = axis
        self.merge_type = merge_type

    def forward(self, x1, x2):
        x1_out = []
        x2_out = []
        for module, x1i, x2i in zip(self._modules.values(), x1, x2):
            y1i, y2i = module(x1i, x2i)
            x1_out.append(y1i)
            x2_out.append(y2i)
        if self.merge_type == "list":
            pass
        elif self.merge_type == "stack":
            x1_out = torch.stack(tuple(x1_out), dim=self.axis)
            x2_out = torch.stack(tuple(x2_out), dim=self.axis)
        elif self.merge_type == "cat":
            x1_out = torch.cat(tuple(x1_out), dim=self.axis)
            x2_out = torch.cat(tuple(x2_out), dim=self.axis)
        elif self.merge_type == "sum":
            x1_out = torch.stack(tuple(x1_out), dim=self.axis).sum(self.axis)
            x2_out = torch.stack(tuple(x2_out), dim=self.axis).sum(self.axis)
        else:
            raise NotImplementedError()
        return x1_out, x2_out
