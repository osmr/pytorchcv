import torch
import torch.nn as nn
from typing import Callable
from common import (lambda_relu, lambda_batchnorm2d, lambda_instancenorm2d, lambda_groupnorm, conv1x1, conv1x1_block,
                    conv3x3_block, conv7x7_block)


class ResBlock(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or None, default lambda_batchnorm2d()
        Lambda-function generator for normalization layer.
    final_activation : function or None, default None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | None = lambda_batchnorm2d(),
                 final_activation: Callable[..., nn.Module | None] | None = None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            normalization=normalization)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=bias,
            normalization=normalization,
            activation=final_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBottleneck(nn.Module):
    """
    ResNet bottleneck block for residual path in ResNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for the second convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for the second convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or None, default lambda_batchnorm2d()
        Lambda-function generator for normalization layer.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    final_activation : function or None, default None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | None = lambda_batchnorm2d(),
                 conv1_stride: bool = False,
                 bottleneck_factor: int = 4,
                 final_activation: Callable[..., nn.Module | None] | None = None):
        super(ResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1),
            bias=bias,
            normalization=normalization)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride),
            padding=padding,
            dilation=dilation,
            bias=bias,
            normalization=normalization)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias,
            normalization=normalization,
            activation=final_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResUnit(nn.Module):
    """
    ResNet unit with residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple(int, int), default 1
        Dilation value for the second convolution layer in bottleneck.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or None, default lambda_batchnorm2d()
        Lambda-function generator for normalization layer.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    final_activation : function or None, default None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | None = lambda_batchnorm2d(),
                 bottleneck: bool = True,
                 conv1_stride: bool = False,
                 final_activation: Callable[..., nn.Module | None] | None = None):
        super(ResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                normalization=normalization,
                conv1_stride=conv1_stride,
                final_activation=final_activation)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                normalization=normalization,
                final_activation=final_activation)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                normalization=normalization,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class BasicEncoder(nn.Module):
    def __init__(self,
                 output_dim,
                 norm_fn,
                 dropout_rate=0.0):
        super(BasicEncoder, self).__init__()
        in_channels = 3
        init_block_channels = 64
        final_block_channels = output_dim
        channels = [[64, 64], [96, 96], [128, 128]]
        bottleneck = False
        conv1_stride = False
        final_activation = lambda_relu()

        if norm_fn == "group":
            normalization = lambda_groupnorm(num_groups=8)
        elif norm_fn == "batch":
            normalization = lambda_batchnorm2d()
        elif norm_fn == "instance":
            normalization = lambda_instancenorm2d()
        elif norm_fn == "none":
            normalization = None
        else:
            assert False

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv7x7_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bias=True,
            normalization=normalization))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bias=True,
                    normalization=normalization,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride,
                    final_activation=final_activation))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", conv1x1(
            in_channels=in_channels,
            out_channels=final_block_channels,
            bias=True))
        if dropout_rate > 0.0:
            self.features.add_module("dropout", nn.Dropout(p=dropout_rate))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.features(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self,
                 output_dim,
                 norm_fn,
                 dropout_rate=0.0):
        super(SmallEncoder, self).__init__()
        in_channels = 3
        init_block_channels = 32
        final_block_channels = output_dim
        channels = [[32, 32], [64, 64], [96, 96]]
        bottleneck = True
        conv1_stride = False
        final_activation = lambda_relu()

        if norm_fn == "group":
            normalization = lambda_groupnorm(num_groups=8)
        elif norm_fn == "batch":
            normalization = lambda_batchnorm2d()
        elif norm_fn == "instance":
            normalization = lambda_instancenorm2d()
        elif norm_fn == "none":
            normalization = None
        else:
            assert False

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv7x7_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bias=True,
            normalization=normalization))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bias=True,
                    normalization=normalization,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride,
                    final_activation=final_activation))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", conv1x1(
            in_channels=in_channels,
            out_channels=final_block_channels,
            bias=True))
        if dropout_rate > 0.0:
            self.features.add_module("dropout", nn.Dropout(p=dropout_rate))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.features(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
