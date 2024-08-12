"""
    Convolution common routines for models in PyTorch.
"""

__all__ = ['ConvBlock1d', 'conv1x1', 'conv3x3', 'depthwise_conv3x3', 'ConvBlock', 'conv1x1_block', 'conv3x3_block',
           'conv5x5_block', 'conv7x7_block', 'dwconv_block', 'dwconv3x3_block', 'dwconv5x5_block', 'dwsconv3x3_block',
           'PreConvBlock', 'pre_conv1x1_block', 'pre_conv3x3_block', 'AsymConvBlock', 'asym_conv3x3_block',
           'DeconvBlock', 'deconv3x3_block']

import torch.nn as nn
from typing import Callable
from .activ import lambda_relu, create_activation_layer
from .norm import lambda_batchnorm1d, lambda_batchnorm2d, create_normalization_layer


class ConvBlock1d(nn.Module):
    """
    Standard 1D convolution block with Batch normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    stride : int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm1d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm1d(),
                 activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu()):
        super(ConvBlock1d, self).__init__()
        self.normalize = (normalization is not None)
        self.activate = (activation is not None)

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.normalize:
            self.bn = create_normalization_layer(
                normalization=normalization,
                num_features=out_channels)
            if self.bn is None:
                self.normalize = False
            else:
                assert isinstance(self.bn, nn.BatchNorm1d)
        if self.activate:
            self.activ = create_activation_layer(activation)
            if self.activ is None:
                self.activate = False

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1(in_channels: int,
            out_channels: int,
            stride: int | tuple[int, int] = 1,
            groups: int = 1,
            bias: bool = False) -> nn.Module:
    """
    Convolution 1x1 layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)


def conv3x3(in_channels: int,
            out_channels: int,
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 1,
            dilation: int | tuple[int, int] = 1,
            groups: int = 1,
            bias: bool = False) -> nn.Module:
    """
    Convolution 3x3 layer.

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
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias)


def depthwise_conv3x3(channels: int,
                      stride: int | tuple[int, int] = 1,
                      padding: int | tuple[int, int] = 1,
                      dilation: int | tuple[int, int] = 1,
                      bias: bool = False) -> nn.Module:
    """
    Depthwise convolution 3x3 layer.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=channels,
        bias=bias)


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 0
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
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                 activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu()):
        super(ConvBlock, self).__init__()
        self.normalize = (normalization is not None)
        self.activate = (activation is not None)
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.normalize:
            # self.bn = nn.BatchNorm2d(
            #     num_features=out_channels,
            #     eps=bn_eps)
            self.bn = create_normalization_layer(
                normalization=normalization,
                num_features=out_channels)
            if self.bn is None:
                self.normalize = False
            else:
                # assert isinstance(self.bn, nn.BatchNorm2d)
                assert isinstance(self.bn, nn.Module)
        if self.activate:
            self.activ = create_activation_layer(activation)
            if self.activ is None:
                self.activate = False
            else:
                assert isinstance(self.activ, nn.Module)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.normalize:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
                  **kwargs) -> nn.Module:
    """
    1x1 version of the standard convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 0
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return ConvBlock(
        kernel_size=1,
        padding=padding,
        **kwargs)


def conv3x3_block(padding: int | tuple[int, int] | tuple[int, int, int, int] = 1,
                  **kwargs) -> nn.Module:
    """
    3x3 version of the standard convolution block.

    Parameters
    ----------
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 1
        Padding value for convolution layer.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return ConvBlock(
        kernel_size=3,
        padding=padding,
        **kwargs)


def conv5x5_block(padding: int | tuple[int, int] | tuple[int, int, int, int] = 2,
                  **kwargs) -> nn.Module:
    """
    5x5 version of the standard convolution block.

    Parameters
    ----------
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 2
        Padding value for convolution layer.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return ConvBlock(
        kernel_size=5,
        padding=padding,
        **kwargs)


def conv7x7_block(padding: int | tuple[int, int] | tuple[int, int, int, int] = 3,
                  **kwargs) -> nn.Module:
    """
    7x7 version of the standard convolution block.

    Parameters
    ----------
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 3
        Padding value for convolution layer.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return ConvBlock(
        kernel_size=7,
        padding=padding,
        **kwargs)


def dwconv_block(out_channels: int,
                 padding: int | tuple[int, int] | tuple[int, int, int, int] = 1,
                 **kwargs) -> nn.Module:
    """
    Depthwise version of the standard convolution block.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 1
        Padding value for convolution layer.
    in_channels : int
        Number of input channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return ConvBlock(
        out_channels=out_channels,
        padding=padding,
        groups=out_channels,
        **kwargs)


def dwconv3x3_block(padding: int | tuple[int, int] | tuple[int, int, int, int] = 1,
                    **kwargs) -> nn.Module:
    """
    3x3 depthwise version of the standard convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 1
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return dwconv_block(
        kernel_size=3,
        padding=padding,
        **kwargs)


def dwconv5x5_block(padding: int | tuple[int, int] | tuple[int, int, int, int] = 2,
                    **kwargs) -> nn.Module:
    """
    5x5 depthwise version of the standard convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 2
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return dwconv_block(
        kernel_size=5,
        padding=padding,
        **kwargs)


class DwsConvBlock(nn.Module):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
    bias : bool, default False
        Whether the layer uses a bias vector.
    dw_normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Normalization function/module for the depthwise convolution block.
    pw_normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Normalization function/module for the pointwise convolution block.
    dw_activation : function or nn.Module or str or None, default lambda_relu()
        Activation function after the depthwise convolution block.
    pw_activation : function or nn.Module or str or None, default lambda_relu()
        Activation function after the pointwise convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int] | tuple[int, int, int, int],
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = False,
                 dw_normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                 pw_normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                 dw_activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu(),
                 pw_activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu()):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            normalization=dw_normalization,
            activation=dw_activation)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            normalization=pw_normalization,
            activation=pw_activation)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


def dwsconv3x3_block(stride: int | tuple[int, int] = 1,
                     padding: int | tuple[int, int] | tuple[int, int, int, int] = 1,
                     **kwargs) -> nn.Module:
    """
    3x3 depthwise separable version of the standard convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 1
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    dw_normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Normalization function/module for the depthwise convolution block.
    pw_normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Normalization function/module for the pointwise convolution block.
    dw_activation : function or nn.Module or str or None, default lambda_relu()
        Activation function after the depthwise convolution block.
    pw_activation : function or nn.Module or str or None, default lambda_relu()
        Activation function after the pointwise convolution block.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return DwsConvBlock(
        kernel_size=3,
        stride=stride,
        padding=padding,
        **kwargs)


class PreConvBlock(nn.Module):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

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
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                 activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu(),
                 return_preact: bool = False):
        super(PreConvBlock, self).__init__()
        self.normalize = (normalization is not None)
        self.activate = (activation is not None)
        self.return_preact = return_preact

        if self.normalize:
            # self.bn = nn.BatchNorm2d(num_features=in_channels)
            self.bn = create_normalization_layer(
                normalization=normalization,
                num_features=in_channels)
            if self.bn is None:
                self.normalize = False
            else:
                assert isinstance(self.bn, nn.BatchNorm2d)
                assert isinstance(self.bn, nn.Module)
        if self.activate:
            # self.activ = nn.ReLU(inplace=True)
            self.activ = create_activation_layer(activation)
            if self.activ is None:
                self.activate = False
            else:
                assert isinstance(self.activ, nn.Module)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

    def forward(self, x):
        if self.normalize:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_conv1x1_block(stride: int | tuple[int, int] = 1,
                      padding: int | tuple[int, int] = 0,
                      **kwargs) -> nn.Module:
    """
    1x1 version of the pre-activated convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int), default 0
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return PreConvBlock(
        kernel_size=1,
        stride=stride,
        padding=padding,
        **kwargs)


def pre_conv3x3_block(stride: int | tuple[int, int] = 1,
                      padding: int | tuple[int, int] = 1,
                      **kwargs) -> nn.Module:
    """
    3x3 version of the pre-activated convolution block.

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
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return PreConvBlock(
        kernel_size=3,
        stride=stride,
        padding=padding,
        **kwargs)


class AsymConvBlock(nn.Module):
    """
    Asymmetric separable convolution block.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
        Whether to use BatchNorm layer (rightwise convolution block).
    lw_normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Normalization function/module for the leftwise convolution block.
    rw_normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Normalization function/module for the rightwise convolution block.
    lw_activation : function or nn.Module or str or None, default lambda_relu()
        Activation function after the leftwise convolution block.
    rw_activation : function or nn.Module or str or None, default lambda_relu()
        Activation function after the rightwise convolution block.
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 padding: int,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 lw_normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                 rw_normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                 lw_activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu(),
                 rw_activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu()):
        super(AsymConvBlock, self).__init__()
        self.lw_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=groups,
            bias=bias,
            normalization=lw_normalization,
            activation=lw_activation)
        self.rw_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            bias=bias,
            normalization=rw_normalization,
            activation=rw_activation)

    def forward(self, x):
        x = self.lw_conv(x)
        x = self.rw_conv(x)
        return x


def asym_conv3x3_block(padding: int = 1,
                       **kwargs) -> nn.Module:
    """
    3x3 asymmetric separable convolution block.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    lw_normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Normalization function/module for the leftwise convolution block.
    rw_normalization : function or nn.Module or None, default lambda_batchnorm2d(eps=1e-5)
        Normalization function/module for the rightwise convolution block.
    lw_activation : function or nn.Module or str or None, default lambda_relu()
        Activation function after the leftwise convolution block.
    rw_activation : function or nn.Module or str or None, default lambda_relu()
        Activation function after the rightwise convolution block.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return AsymConvBlock(
        kernel_size=3,
        padding=padding,
        **kwargs)


class DeconvBlock(nn.Module):
    """
    Deconvolution block with batch normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    stride : int or tuple(int, int)
        Strides of the deconvolution.
    padding : int or tuple(int, int)
        Padding value for deconvolution layer.
    ext_padding : tuple(int, int, int, int) or None, default None
        Extra padding value for deconvolution layer.
    out_padding : int or tuple(int, int), default 0
        Output padding value for deconvolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for deconvolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 ext_padding: tuple[int, int, int, int] | None = None,
                 out_padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                 activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu()):
        super(DeconvBlock, self).__init__()
        self.normalize = (normalization is not None)
        self.activate = (activation is not None)
        self.use_pad = (ext_padding is not None)

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=ext_padding)
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=out_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.normalize:
            # self.bn = nn.BatchNorm2d(
            #     num_features=out_channels,
            #     eps=bn_eps)
            self.bn = create_normalization_layer(
                normalization=normalization,
                num_features=out_channels)
            if self.bn is None:
                self.normalize = False
            else:
                assert isinstance(self.bn, nn.BatchNorm2d)
        if self.activate:
            self.activ = create_activation_layer(activation)
            if self.activ is None:
                self.activate = False

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.normalize:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def deconv3x3_block(padding: int | tuple[int, int] = 1,
                    out_padding: int | tuple[int, int] = 1,
                    **kwargs) -> nn.Module:
    """
    3x3 version of the deconvolution block with batch normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the deconvolution.
    padding : int or tuple(int, int), default 1
        Padding value for deconvolution layer.
    ext_padding : tuple/list of 4 int, default None
        Extra padding value for deconvolution layer.
    out_padding : int or tuple(int, int), default 1
        Output padding value for deconvolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for deconvolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return DeconvBlock(
        kernel_size=3,
        padding=padding,
        out_padding=out_padding,
        **kwargs)
