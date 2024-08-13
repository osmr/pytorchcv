"""
    Attention common routines for models in PyTorch.
"""

__all__ = ['round_channels', 'SEBlock', 'SABlock', 'SAConvBlock', 'saconv3x3_block']

import torch
import torch.nn as nn
from typing import Callable
from .activ import lambda_relu, lambda_sigmoid, create_activation_layer
from .norm import lambda_batchnorm2d, create_normalization_layer
from .conv import conv1x1, ConvBlock


def round_channels(channels: int | float,
                   divisor: int = 8) -> int:
    """
    Round weighted channel number (make divisible operation).

    Parameters
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.

    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    mid_channels : int or None, default None
        Number of middle channels.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    mid_activation : function or nn.Module or str, default lambda_relu()
        Activation function after the first convolution.
    out_activation : function or nn.Module or str, default lambda_sigmoid()
        Activation function after the last convolution.
    """
    def __init__(self,
                 channels: int,
                 reduction: int = 16,
                 mid_channels: int | None = None,
                 round_mid: bool = False,
                 use_conv: bool = True,
                 mid_activation: Callable[..., nn.Module] | nn.Module | str = lambda_relu(),
                 out_activation: Callable[..., nn.Module] | nn.Module | str = lambda_sigmoid()):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_channels,
                bias=True)
        else:
            self.fc1 = nn.Linear(
                in_features=channels,
                out_features=mid_channels)
        self.activ = create_activation_layer(mid_activation)
        if use_conv:
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels,
                bias=True)
        else:
            self.fc2 = nn.Linear(
                in_features=mid_channels,
                out_features=channels)
        self.sigmoid = create_activation_layer(out_activation)

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x


class SABlock(nn.Module):
    """
    Split-Attention block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    groups : int
        Number of channel groups (cardinality, without radix).
    radix : int
        Number of splits within a cardinal group.
    reduction : int, default 4
        Squeeze reduction value.
    min_channels : int, default 32
        Minimal number of squeezed channels.
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    normalization : function or nn.Module, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    """
    def __init__(self,
                 out_channels: int,
                 groups: int,
                 radix: int,
                 reduction: int = 4,
                 min_channels: int = 32,
                 use_conv: bool = True,
                 normalization: Callable[..., nn.Module] | nn.Module = lambda_batchnorm2d()):
        super(SABlock, self).__init__()
        self.groups = groups
        self.radix = radix
        self.use_conv = use_conv
        in_channels = out_channels * radix
        mid_channels = max(in_channels // reduction, min_channels)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = conv1x1(
                in_channels=out_channels,
                out_channels=mid_channels,
                bias=True)
        else:
            self.fc1 = nn.Linear(
                in_features=out_channels,
                out_features=mid_channels)
        # self.bn = nn.BatchNorm2d(
        #     num_features=mid_channels,
        #     eps=bn_eps)
        self.bn = create_normalization_layer(
            normalization=normalization,
            num_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        if use_conv:
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=in_channels,
                bias=True)
        else:
            self.fc2 = nn.Linear(
                in_features=mid_channels,
                out_features=in_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        x = x.view(batch, self.radix, channels // self.radix, height, width)
        w = x.sum(dim=1)
        w = self.pool(w)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.bn(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = w.view(batch, self.groups, self.radix, -1)
        w = torch.transpose(w, 1, 2).contiguous()
        w = self.softmax(w)
        w = w.view(batch, self.radix, -1, 1, 1)
        x = x * w
        x = x.sum(dim=1)
        return x


class SAConvBlock(nn.Module):
    """
    Split-Attention convolution block from 'ResNeSt: Split-Attention Networks,' https://arxiv.org/abs/2004.08955.

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
    padding : int or tuple(int, int) or tuple(int, int, int, int)
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    radix : int, default 2
        Number of splits within a cardinal group.
    reduction : int, default 4
        Squeeze reduction value.
    min_channels : int, default 32
        Minimal number of squeezed channels.
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int] | tuple[int, int, int, int],
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                 activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu(),
                 radix: int = 2,
                 reduction: int = 4,
                 min_channels: int = 32,
                 use_conv: bool = True):
        super(SAConvBlock, self).__init__()
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=(out_channels * radix),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=(groups * radix),
            bias=bias,
            normalization=normalization,
            activation=activation)
        self.att = SABlock(
            out_channels=out_channels,
            groups=groups,
            radix=radix,
            reduction=reduction,
            min_channels=min_channels,
            use_conv=use_conv,
            normalization=normalization)

    def forward(self, x):
        x = self.conv(x)
        x = self.att(x)
        return x


def saconv3x3_block(stride: int | tuple[int, int] = 1,
                    padding: int | tuple[int, int] = 1,
                    **kwargs) -> nn.Module:
    """
    3x3 version of the Split-Attention convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for convolution layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return SAConvBlock(
        kernel_size=3,
        stride=stride,
        padding=padding,
        **kwargs)
