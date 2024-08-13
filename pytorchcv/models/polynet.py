"""
    PolyNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'PolyNet: A Pursuit of Structural Diversity in Very Deep Networks,'
    https://arxiv.org/abs/1611.05725.
"""

__all__ = ['PolyNet', 'polynet']

import os
import torch.nn as nn
from typing import Callable
from .common.conv import ConvBlock, conv1x1_block, conv3x3_block
from .common.arch import Concurrent, ParametricSequential, ParametricConcurrent


class PolyConv(nn.Module):
    """
    PolyNet specific convolution block. A block that is used inside poly-N (poly-2, poly-3, and so on) modules.
    The Convolution layer is shared between all Inception blocks inside a poly-N module. BatchNorm layers are not
    shared between Inception blocks and therefore the number of BatchNorm layers is equal to the number of Inception
    blocks inside a poly-N module.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int)
        Padding value for convolution layer.
    num_blocks : int
        Number of blocks (BatchNorm layers).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 num_blocks: int):
        super(PolyConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bns = nn.ModuleList()
        for i in range(num_blocks):
            self.bns.append(nn.BatchNorm2d(num_features=out_channels))
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x, index):
        x = self.conv(x)
        x = self.bns[index](x)
        x = self.activ(x)
        return x


def poly_conv1x1(in_channels: int,
                 out_channels: int,
                 num_blocks: int):
    """
    1x1 version of the PolyNet specific convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_blocks : int
        Number of blocks (BatchNorm layers).
    """
    return PolyConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        num_blocks=num_blocks)


class MaxPoolBranch(nn.Module):
    """
    PolyNet specific max pooling branch block.
    """
    def __init__(self):
        super(MaxPoolBranch, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.pool(x)
        return x


class Conv1x1Branch(nn.Module):
    """
    PolyNet specific convolutional 1x1 branch block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(Conv1x1Branch, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv3x3Branch(nn.Module):
    """
    PolyNet specific convolutional 3x3 branch block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(Conv3x3Branch, self).__init__()
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvSeqBranch(nn.Module):
    """
    PolyNet specific convolutional sequence branch block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : tuple(int, ...)
        List of numbers of output channels.
    kernel_size_list : tuple(int or tuple(int, int), ...)
        List of convolution window sizes.
    strides_list : tuple(int or tuple(int, int), ...)
        List of strides of the convolution.
    padding_list : tuple(int or tuple(int, int), ...)
        List of padding values for convolution layers.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels_list: tuple[int, ...],
                 kernel_size_list: tuple[int | tuple[int, int], ...],
                 strides_list: tuple[int | tuple[int, int], ...],
                 padding_list: tuple[int | tuple[int, int], ...]):
        super(ConvSeqBranch, self).__init__()
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv_list(x)
        return x


class PolyConvSeqBranch(nn.Module):
    """
    PolyNet specific convolutional sequence branch block with internal PolyNet specific convolution blocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : tuple(int, ...)
        List of numbers of output channels.
    kernel_size_list : tuple(int or tuple(int, int), ...)
        List of convolution window sizes.
    strides_list : tuple(int or tuple(int, int), ...)
        List of strides of the convolution.
    padding_list : tuple(int or tuple(int, int), ...)
        List of padding values for convolution layers.
    num_blocks : int
        Number of blocks for PolyConv.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels_list: tuple[int, ...],
                 kernel_size_list: tuple[int | tuple[int, int], ...],
                 strides_list: tuple[int | tuple[int, int], ...],
                 padding_list: tuple[int | tuple[int, int], ...],
                 num_blocks: int):
        super(PolyConvSeqBranch, self).__init__()
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = ParametricSequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), PolyConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                num_blocks=num_blocks))
            in_channels = out_channels

    def forward(self, x, index):
        x = self.conv_list(x, index=index)
        return x


class TwoWayABlock(nn.Module):
    """
    PolyNet type Inception-A block.
    """
    def __init__(self):
        super(TwoWayABlock, self).__init__()
        in_channels = 384

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(32, 48, 64),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(32, 32),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1)))
        self.branches.add_module("branch3", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=32))
        self.conv = conv1x1_block(
            in_channels=128,
            out_channels=in_channels,
            activation=None)

    def forward(self, x):
        x = self.branches(x)
        x = self.conv(x)
        return x


class TwoWayBBlock(nn.Module):
    """
    PolyNet type Inception-B block.
    """
    def __init__(self):
        super(TwoWayBBlock, self).__init__()
        in_channels = 1152

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(128, 160, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0))))
        self.branches.add_module("branch2", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=192))
        self.conv = conv1x1_block(
            in_channels=384,
            out_channels=in_channels,
            activation=None)

    def forward(self, x):
        x = self.branches(x)
        x = self.conv(x)
        return x


class TwoWayCBlock(nn.Module):
    """
    PolyNet type Inception-C block.
    """
    def __init__(self):
        super(TwoWayCBlock, self).__init__()
        in_channels = 2048

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 3), (3, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 1), (1, 0))))
        self.branches.add_module("branch2", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=192))
        self.conv = conv1x1_block(
            in_channels=448,
            out_channels=in_channels,
            activation=None)

    def forward(self, x):
        x = self.branches(x)
        x = self.conv(x)
        return x


class PolyPreBBlock(nn.Module):
    """
    PolyNet type PolyResidual-Pre-B block.

    Parameters
    ----------
    num_blocks : int
        Number of blocks (BatchNorm layers).
    """
    def __init__(self,
                 num_blocks: int):
        super(PolyPreBBlock, self).__init__()
        in_channels = 1152

        self.branches = ParametricConcurrent()
        self.branches.add_module("branch1", PolyConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(128, 160, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            num_blocks=num_blocks))
        self.branches.add_module("branch2", poly_conv1x1(
            in_channels=in_channels,
            out_channels=192,
            num_blocks=num_blocks))

    def forward(self, x, index):
        x = self.branches(x, index=index)
        return x


class PolyPreCBlock(nn.Module):
    """
    PolyNet type PolyResidual-Pre-C block.

    Parameters
    ----------
    num_blocks : int
        Number of blocks (BatchNorm layers).
    """
    def __init__(self,
                 num_blocks: int):
        super(PolyPreCBlock, self).__init__()
        in_channels = 2048

        self.branches = ParametricConcurrent()
        self.branches.add_module("branch1", PolyConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 3), (3, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 1), (1, 0)),
            num_blocks=num_blocks))
        self.branches.add_module("branch2", poly_conv1x1(
            in_channels=in_channels,
            out_channels=192,
            num_blocks=num_blocks))

    def forward(self, x, index):
        x = self.branches(x, index=index)
        return x


def poly_res_b_block():
    """
    PolyNet type PolyResidual-Res-B block.
    """
    return conv1x1_block(
        in_channels=384,
        out_channels=1152,
        stride=1,
        activation=None)


def poly_res_c_block():
    """
    PolyNet type PolyResidual-Res-C block.
    """
    return conv1x1_block(
        in_channels=448,
        out_channels=2048,
        stride=1,
        activation=None)


class MultiResidual(nn.Module):
    """
    Base class for constructing N-way modules (2-way, 3-way, and so on). Actually it is for 2-way modules.

    Parameters
    ----------
    scale : float
        Scale value for each residual branch.
    res_block : type(nn.Module)
        Residual branch block.
    num_blocks : int
        Number of residual branches.
    """
    def __init__(self,
                 scale: float,
                 res_block: type[nn.Module],
                 num_blocks: int):
        super(MultiResidual, self).__init__()
        assert (num_blocks >= 1)
        self.scale = scale

        self.res_blocks = nn.ModuleList([res_block() for _ in range(num_blocks)])
        self.activ = nn.ReLU(inplace=False)

    def forward(self, x):
        out = x
        for res_block in self.res_blocks:
            out = out + self.scale * res_block(x)
        out = self.activ(out)
        return out


class PolyResidual(nn.Module):
    """
    The other base class for constructing N-way poly-modules. Actually it is for 3-way poly-modules.

    Parameters
    ----------
    scale : float
        Scale value for each residual branch.
    res_block : type(nn.Module)
        Residual branch block.
    num_blocks : int
        Number of residual branches.
    pre_block : type(nn.Module)
        Preliminary block.
    """
    def __init__(self,
                 scale: float,
                 res_block: type[nn.Module],
                 num_blocks: int,
                 pre_block: type[nn.Module]):
        super(PolyResidual, self).__init__()
        assert (num_blocks >= 1)
        self.scale = scale

        self.pre_block = pre_block(num_blocks=num_blocks)
        self.res_blocks = nn.ModuleList([res_block() for _ in range(num_blocks)])
        self.activ = nn.ReLU(inplace=False)

    def forward(self, x):
        out = x
        for index, res_block in enumerate(self.res_blocks):
            x = self.pre_block(x, index)
            x = res_block(x)
            out = out + self.scale * x
            x = self.activ(x)
        out = self.activ(out)
        return out


class PolyBaseUnit(nn.Module):
    """
    PolyNet unit base class.

    Parameters
    ----------
    two_way_scale : float
        Scale value for 2-way stage.
    two_way_block : type(nn.Module)
        Residual branch block for 2-way-stage.
    poly_scale : float, default 0.0
        Scale value for 2-way stage.
    poly_res_block : type(nn.Module) or function or None, default None
        Residual branch block for poly-stage.
    poly_pre_block : type(nn.Module) or function or None, default None
        Preliminary branch block for poly-stage.
    """
    def __init__(self,
                 two_way_scale: float,
                 two_way_block: type[nn.Module],
                 poly_scale: float = 0.0,
                 poly_res_block: type[nn.Module] | Callable[..., nn.Module] | None = None,
                 poly_pre_block: type[nn.Module] | Callable[..., nn.Module] | None = None):
        super(PolyBaseUnit, self).__init__()

        if poly_res_block is not None:
            assert (poly_scale != 0.0)
            assert (poly_pre_block is not None)
            self.poly = PolyResidual(
                scale=poly_scale,
                res_block=poly_res_block,
                num_blocks=3,
                pre_block=poly_pre_block)
        else:
            assert (poly_scale == 0.0)
            assert (poly_pre_block is None)
            self.poly = None
        self.twoway = MultiResidual(
            scale=two_way_scale,
            res_block=two_way_block,
            num_blocks=2)

    def forward(self, x):
        if self.poly is not None:
            x = self.poly(x)
        x = self.twoway(x)
        return x


class PolyAUnit(PolyBaseUnit):
    """
    PolyNet type A unit.

    Parameters
    ----------
    two_way_scale : float
        Scale value for 2-way stage.
    poly_scale : float
        Scale value for 2-way stage.
    """
    def __init__(self,
                 two_way_scale: float,
                 poly_scale: float = 0.0):
        super(PolyAUnit, self).__init__(
            two_way_scale=two_way_scale,
            two_way_block=TwoWayABlock)
        assert (poly_scale == 0.0)


class PolyBUnit(PolyBaseUnit):
    """
    PolyNet type B unit.

    Parameters
    ----------
    two_way_scale : float
        Scale value for 2-way stage.
    poly_scale : float
        Scale value for 2-way stage.
    """
    def __init__(self,
                 two_way_scale: float,
                 poly_scale: float):
        super(PolyBUnit, self).__init__(
            two_way_scale=two_way_scale,
            two_way_block=TwoWayBBlock,
            poly_scale=poly_scale,
            poly_res_block=poly_res_b_block,
            poly_pre_block=PolyPreBBlock)


class PolyCUnit(PolyBaseUnit):
    """
    PolyNet type C unit.

    Parameters
    ----------
    two_way_scale : float
        Scale value for 2-way stage.
    poly_scale : float
        Scale value for 2-way stage.
    """
    def __init__(self,
                 two_way_scale: float,
                 poly_scale: float):
        super(PolyCUnit, self).__init__(
            two_way_scale=two_way_scale,
            two_way_block=TwoWayCBlock,
            poly_scale=poly_scale,
            poly_res_block=poly_res_c_block,
            poly_pre_block=PolyPreCBlock)


class ReductionAUnit(nn.Module):
    """
    PolyNet type Reduction-A unit.
    """
    def __init__(self):
        super(ReductionAUnit, self).__init__()
        in_channels = 384

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256, 384),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,)))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionBUnit(nn.Module):
    """
    PolyNet type Reduction-B unit.
    """
    def __init__(self):
        super(ReductionBUnit, self).__init__()
        in_channels = 1152

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256, 256),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0)))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 384),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0)))
        self.branches.add_module("branch4", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class PolyBlock3a(nn.Module):
    """
    PolyNet type Mixed-3a block.
    """
    def __init__(self):
        super(PolyBlock3a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", MaxPoolBranch())
        self.branches.add_module("branch2", Conv3x3Branch(
            in_channels=64,
            out_channels=96))

    def forward(self, x):
        x = self.branches(x)
        return x


class PolyBlock4a(nn.Module):
    """
    PolyNet type Mixed-4a block.
    """
    def __init__(self):
        super(PolyBlock4a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 0)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 64, 64, 96),
            kernel_size_list=(1, (7, 1), (1, 7), 3),
            strides_list=(1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), 0)))

    def forward(self, x):
        x = self.branches(x)
        return x


class PolyBlock5a(nn.Module):
    """
    PolyNet type Mixed-5a block.
    """
    def __init__(self):
        super(PolyBlock5a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", MaxPoolBranch())
        self.branches.add_module("branch2", Conv3x3Branch(
            in_channels=192,
            out_channels=192))

    def forward(self, x):
        x = self.branches(x)
        return x


class PolyInitBlock(nn.Module):
    """
    PolyNet specific initial block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels: int):
        super(PolyInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=32,
            stride=2,
            padding=0)
        self.conv2 = conv3x3_block(
            in_channels=32,
            out_channels=32,
            padding=0)
        self.conv3 = conv3x3_block(
            in_channels=32,
            out_channels=64)
        self.block1 = PolyBlock3a()
        self.block2 = PolyBlock4a()
        self.block3 = PolyBlock5a()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class PolyNet(nn.Module):
    """
    PolyNet model from 'PolyNet: A Pursuit of Structural Diversity in Very Deep Networks,'
    https://arxiv.org/abs/1611.05725.

    Parameters
    ----------
    two_way_scales : list(list(float))
        Two way scale values for each normal unit.
    poly_scales : list(list(float))
        Three way scale values for each normal unit.
    dropout_rate : float, default 0.2
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (331, 331)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 two_way_scales: list[list[float]],
                 poly_scales: list[list[float]],
                 dropout_rate: float = 0.2,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (331, 331),
                 num_classes: int = 1000):
        super(PolyNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        normal_units = [PolyAUnit, PolyBUnit, PolyCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = nn.Sequential()
        self.features.add_module("init_block", PolyInitBlock(
            in_channels=in_channels))

        for i, (two_way_scales_per_stage, poly_scales_per_stage) in enumerate(zip(two_way_scales, poly_scales)):
            stage = nn.Sequential()
            for j, (two_way_scale, poly_scale) in enumerate(zip(two_way_scales_per_stage, poly_scales_per_stage)):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                    stage.add_module("unit{}".format(j + 1), unit())
                else:
                    unit = normal_units[i]
                    stage.add_module("unit{}".format(j + 1), unit(
                        two_way_scale=two_way_scale,
                        poly_scale=poly_scale))
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=9,
            stride=1))

        self.output = nn.Sequential()
        self.output.add_module("dropout", nn.Dropout(p=dropout_rate))
        self.output.add_module("fc", nn.Linear(
            in_features=2048,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_polynet(model_name: str | None = None,
                pretrained: bool = False,
                root: str = os.path.join("~", ".torch", "models"),
                **kwargs) -> nn.Module:
    """
    Create PolyNet model with specific parameters.

    Parameters
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    two_way_scales = [
        [1.000000, 0.992308, 0.984615, 0.976923, 0.969231, 0.961538, 0.953846, 0.946154, 0.938462, 0.930769],
        [0.000000, 0.915385, 0.900000, 0.884615, 0.869231, 0.853846, 0.838462, 0.823077, 0.807692, 0.792308, 0.776923],
        [0.000000, 0.761538, 0.746154, 0.730769, 0.715385, 0.700000]]
    poly_scales = [
        [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        [0.000000, 0.923077, 0.907692, 0.892308, 0.876923, 0.861538, 0.846154, 0.830769, 0.815385, 0.800000, 0.784615],
        [0.000000, 0.769231, 0.753846, 0.738462, 0.723077, 0.707692]]

    net = PolyNet(
        two_way_scales=two_way_scales,
        poly_scales=poly_scales,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .common.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def polynet(**kwargs) -> nn.Module:
    """
    PolyNet model from 'PolyNet: A Pursuit of Structural Diversity in Very Deep Networks,'
    https://arxiv.org/abs/1611.05725.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_polynet(
        model_name="polynet",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        polynet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != polynet or weight_count == 95366600)

        x = torch.randn(1, 3, 331, 331)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
