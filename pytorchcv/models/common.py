"""
    Common routines for models in PyTorch.
"""

__all__ = ['round_channels', 'Identity', 'BreakBlock', 'Swish', 'HSigmoid', 'HSwish', 'lambda_relu', 'lambda_relu6',
           'lambda_prelu', 'lambda_leakyrelu', 'lambda_sigmoid', 'lambda_hsigmoid', 'lambda_swish', 'lambda_hswish',
           'lambda_batchnorm1d', 'lambda_batchnorm2d', 'create_activation_layer', 'create_normalization_layer',
           'SelectableDense', 'DenseBlock', 'ConvBlock1d', 'conv1x1', 'conv3x3', 'depthwise_conv3x3', 'ConvBlock',
           'conv1x1_block', 'conv3x3_block', 'conv5x5_block', 'conv7x7_block', 'dwconv_block', 'dwconv3x3_block',
           'dwconv5x5_block', 'dwsconv3x3_block', 'PreConvBlock', 'pre_conv1x1_block', 'pre_conv3x3_block',
           'AsymConvBlock', 'asym_conv3x3_block', 'DeconvBlock', 'deconv3x3_block', 'NormActivation',
           'InterpolationBlock', 'ChannelShuffle', 'ChannelShuffle2', 'SEBlock', 'SABlock', 'SAConvBlock',
           'saconv3x3_block', 'DucBlock', 'IBN', 'DualPathSequential', 'Concurrent', 'SequentialConcurrent',
           'ParametricSequential', 'ParametricConcurrent', 'Hourglass', 'SesquialteralHourglass',
           'MultiOutputSequential', 'ParallelConcurent', 'DualPathParallelConcurent', 'Flatten', 'HeatmapMaxDetBlock']

import math
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Callable


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


class Identity(nn.Module):
    """
    Identity block.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        return "{name}()".format(name=self.__class__.__name__)


class BreakBlock(nn.Module):
    """
    Break coonnection block for hourglass.
    """
    def __init__(self):
        super(BreakBlock, self).__init__()

    def forward(self, x):
        return None

    def __repr__(self):
        return "{name}()".format(name=self.__class__.__name__)


class Swish(nn.Module):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters
    ----------
    inplace : bool, default False
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace: bool = False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


def lambda_relu(inplace: bool = True) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.ReLU activation layer.

    Parameters
    ----------
    inplace : bool, default true
        Whether to do the operation in-place.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.ReLU(inplace=inplace)


def lambda_relu6(inplace: bool = True) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.ReLU6 activation layer.

    Parameters
    ----------
    inplace : bool, default true
        Whether to do the operation in-place.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.ReLU6(inplace=inplace)


def lambda_prelu(num_parameters: int = 1) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.PReLU activation layer.

    Parameters
    ----------
    num_parameters : int, default 1
        Number of `a` to learn. There is only two values are legitimate: 1, or the number of channels at input.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.PReLU(num_parameters=num_parameters)


def lambda_leakyrelu(negative_slope: float = 1e-2,
                     inplace: bool = True) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.LeakyReLU activation layer.

    Parameters
    ----------
    negative_slope : float, default 1e-2
        Slope coefficient controls the angle of the negative slope (which is used for negative input values).
    inplace : bool, default true
        Whether to do the operation in-place.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.LeakyReLU(
        negative_slope=negative_slope,
        inplace=inplace)


def lambda_sigmoid() -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.Sigmoid activation layer.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.Sigmoid()


def lambda_hsigmoid() -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for HSigmoid activation layer.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: HSigmoid()


def lambda_swish() -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for Swish activation layer.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: Swish()


def lambda_hswish(inplace: bool = True) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for HSwish activation layer.

    Parameters
    ----------
    inplace : bool, default true
        Whether to do the operation in-place.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: HSwish(inplace=inplace)


def lambda_batchnorm1d(eps: float = 1e-5) -> Callable[[int], nn.Module]:
    """
    Create lambda-function generator for nn.BatchNorm1d normalization layer.

    Parameters
    ----------
    eps : float, default 1e-5
        Batch-norm epsilon.

    Returns
    -------
    function
        Desired function.
    """
    return lambda num_features: nn.BatchNorm1d(
        num_features=num_features,
        eps=eps)


def lambda_batchnorm2d(eps: float = 1e-5) -> Callable[[int], nn.Module]:
    """
    Create lambda-function generator for nn.BatchNorm2d normalization layer.

    Parameters
    ----------
    eps : float, default 1e-5
        Batch-norm epsilon.

    Returns
    -------
    function
        Desired function.
    """
    return lambda num_features: nn.BatchNorm2d(
        num_features=num_features,
        eps=eps)


def create_activation_layer(activation: Callable[..., nn.Module | None] | nn.Module | str) -> nn.Module | None:
    """
    Create activation layer from lambda-function generator or module.

    Parameters
    ----------
    activation : function or nn.Module or str
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module or None
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "swish":
            return Swish()
        elif activation == "hswish":
            return HSwish(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "hsigmoid":
            return HSigmoid()
        elif activation == "identity":
            return Identity()
        else:
            raise NotImplementedError()
    else:
        assert isinstance(activation, nn.Module)
        return activation


def create_normalization_layer(normalization: Callable[..., nn.Module | None] | nn.Module,
                               **kwargs) -> nn.Module | None:
    """
    Create normalization layer from lambda-function generator or module.

    Parameters
    ----------
    normalization : function or nn.Module
        Lambda-function generator or module for normalization layer.

    Returns
    -------
    nn.Module or None
        Normalization layer.
    """
    assert (normalization is not None)
    if isfunction(normalization):
        return normalization(**kwargs)
    else:
        assert isinstance(normalization, nn.Module)
        return normalization


class SelectableDense(nn.Module):
    """
    Selectable dense layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, default False
        Whether the layer uses a bias vector.
    num_options : int, default 1
        Number of selectable options.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 num_options: int = 1):
        super(SelectableDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.num_options = num_options
        self.weight = Parameter(torch.Tensor(num_options, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(num_options, out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, indices):
        weight = torch.index_select(self.weight, dim=0, index=indices)
        x = x.unsqueeze(-1)
        x = weight.bmm(x)
        x = x.squeeze(dim=-1)
        if self.use_bias:
            bias = torch.index_select(self.bias, dim=0, index=indices)
            x += bias
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, num_options={}".format(
            self.in_features, self.out_features, self.use_bias, self.num_options)


class DenseBlock(nn.Module):
    """
    Standard dense block with Batch normalization and activation.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm1d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm1d(),
                 activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu()):
        super(DenseBlock, self).__init__()
        self.normalize = (normalization is not None)
        self.activate = (activation is not None)

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias)
        if self.normalize:
            # self.bn = nn.BatchNorm1d(
            #     num_features=out_features,
            #     eps=bn_eps)
            self.bn = create_normalization_layer(
                normalization=normalization,
                num_features=out_features)
            if self.bn is None:
                self.normalize = False
            else:
                assert isinstance(self.bn, nn.BatchNorm1d)
        if self.activate:
            self.activ = create_activation_layer(activation)
            if self.activ is None:
                self.activate = False

    def forward(self, x):
        x = self.fc(x)
        if self.normalize:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


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
            # self.bn = nn.BatchNorm1d(
            #     num_features=out_channels,
            #     eps=bn_eps)
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
                assert isinstance(self.bn, nn.BatchNorm2d)
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


class NormActivation(nn.Module):
    """
    Activation block with preliminary batch normalization. It's used by itself as the final block in PreResNet.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    normalization : function or nn.Module, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str, default lambda_relu()
        Lambda-function generator or module for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 normalization: Callable[..., nn.Module] | nn.Module = lambda_batchnorm2d(),
                 activation: Callable[..., nn.Module] | nn.Module | str = lambda_relu()):
        super(NormActivation, self).__init__()
        # self.bn = nn.BatchNorm2d(
        #     num_features=in_channels,
        #     eps=bn_eps)
        self.bn = create_normalization_layer(
            normalization=normalization,
            num_features=in_channels)
        assert (self.bn is not None)
        assert isinstance(self.bn, nn.BatchNorm2d)
        assert isinstance(self.bn, nn.Module)
        self.activ = create_activation_layer(activation)
        assert (self.activ is not None)
        assert isinstance(self.activ, nn.Module)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class InterpolationBlock(nn.Module):
    """
    Interpolation upsampling block.

    Parameters
    ----------
    scale_factor : int or None
        Multiplier for spatial size.
    out_size : tuple(int, int) or None, default None
        Spatial size of the output tensor for the bilinear interpolation operation.
    mode : str, default 'bilinear'
        Algorithm used for upsampling.
    align_corners : bool or None, default True
        Whether to align the corner pixels of the input and output tensors.
    up : bool, default True
        Whether to upsample or downsample.
    """
    def __init__(self,
                 scale_factor: int | None,
                 out_size: tuple[int, int] | None = None,
                 mode: str = "bilinear",
                 align_corners: bool | None = True,
                 up: bool = True):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor
        self.out_size = out_size
        self.mode = mode
        self.align_corners = align_corners
        self.up = up

    def forward(self, x, size=None):
        if (self.mode == "bilinear") or (size is not None):
            out_size = self.calc_out_size(x) if size is None else size
            return F.interpolate(
                input=x,
                size=out_size,
                mode=self.mode,
                align_corners=self.align_corners)
        else:
            return F.interpolate(
                input=x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=self.align_corners)

    def calc_out_size(self, x):
        if self.out_size is not None:
            return self.out_size
        if self.up:
            return tuple(s * self.scale_factor for s in x.shape[2:])
        else:
            return tuple(s // self.scale_factor for s in x.shape[2:])

    def __repr__(self):
        s = "{name}(scale_factor={scale_factor}, out_size={out_size}, mode={mode}, align_corners={align_corners}, up={up})" # noqa
        return s.format(
            name=self.__class__.__name__,
            scale_factor=self.scale_factor,
            out_size=self.out_size,
            mode=self.mode,
            align_corners=self.align_corners,
            up=self.up)

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        if self.mode == "bilinear":
            num_flops = 9 * x.numel()
        else:
            num_flops = 4 * x.numel()
        num_macs = 0
        return num_flops, num_macs


def channel_shuffle(x: torch.Tensor,
                    groups: int) -> torch.Tensor:
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    groups : int
        Number of groups.

    Returns
    -------
    torch.Tensor
        Resulted tensor.
    """
    batch, channels, height, width = x.size()
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels: int,
                 groups: int):
        super(ChannelShuffle, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

    def __repr__(self):
        s = "{name}(groups={groups})"
        return s.format(
            name=self.__class__.__name__,
            groups=self.groups)


def channel_shuffle2(x: torch.Tensor,
                     groups: int) -> torch.Tensor:
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083. The alternative version.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    groups : int
        Number of groups.

    Returns
    -------
    torch.Tensor
        Resulted tensor.
    """
    batch, channels, height, width = x.size()
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle2(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    The alternative version.

    Parameters
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels: int,
                 groups: int):
        super(ChannelShuffle2, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle2(x, self.groups)


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


class DucBlock(nn.Module):
    """
    Dense Upsampling Convolution (DUC) block from 'Understanding Convolution for Semantic Segmentation,'
    https://arxiv.org/abs/1702.08502.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    scale_factor : int
        Multiplier for spatial size.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int):
        super(DucBlock, self).__init__()
        mid_channels = (scale_factor * scale_factor) * out_channels

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.pix_shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pix_shuffle(x)
        return x


class IBN(nn.Module):
    """
    Instance-Batch Normalization block from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters
    ----------
    channels : int
        Number of channels.
    first_fraction : float, default 0.5
        The first fraction of channels for normalization.
    inst_first : bool, default True
        Whether instance normalization be on the first part of channels.
    """
    def __init__(self,
                 channels: int,
                 first_fraction: float = 0.5,
                 inst_first: bool = True):
        super(IBN, self).__init__()
        self.inst_first = inst_first
        h1_channels = int(math.floor(channels * first_fraction))
        h2_channels = channels - h1_channels
        self.split_sections = [h1_channels, h2_channels]

        if self.inst_first:
            self.inst_norm = nn.InstanceNorm2d(
                num_features=h1_channels,
                affine=True)
            self.batch_norm = nn.BatchNorm2d(num_features=h2_channels)
        else:
            self.batch_norm = nn.BatchNorm2d(num_features=h1_channels)
            self.inst_norm = nn.InstanceNorm2d(
                num_features=h2_channels,
                affine=True)

    def forward(self, x):
        x1, x2 = torch.split(x, split_size_or_sections=self.split_sections, dim=1)
        if self.inst_first:
            x1 = self.inst_norm(x1.contiguous())
            x2 = self.batch_norm(x2.contiguous())
        else:
            x1 = self.batch_norm(x1.contiguous())
            x2 = self.inst_norm(x2.contiguous())
        x = torch.cat((x1, x2), dim=1)
        return x


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


class Flatten(nn.Module):
    """
    Simple flatten module.
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class HeatmapMaxDetBlock(nn.Module):
    """
    Heatmap maximum detector block (for human pose estimation task).
    """
    def __init__(self):
        super(HeatmapMaxDetBlock, self).__init__()

    def forward(self, x):
        heatmap = x
        vector_dim = 2
        batch = heatmap.shape[0]
        channels = heatmap.shape[1]
        in_size = x.shape[2:]
        heatmap_vector = heatmap.view(batch, channels, -1)
        scores, indices = heatmap_vector.max(dim=vector_dim, keepdims=True)
        scores_mask = (scores > 0.0).float()
        pts_x = (indices % in_size[1]) * scores_mask
        pts_y = (indices // in_size[1]) * scores_mask
        pts = torch.cat((pts_x, pts_y, scores), dim=vector_dim)
        for b in range(batch):
            for k in range(channels):
                hm = heatmap[b, k, :, :]
                px = int(pts[b, k, 0])
                py = int(pts[b, k, 1])
                if (0 < px < in_size[1] - 1) and (0 < py < in_size[0] - 1):
                    pts[b, k, 0] += (hm[py, px + 1] - hm[py, px - 1]).sign() * 0.25
                    pts[b, k, 1] += (hm[py + 1, px] - hm[py - 1, px]).sign() * 0.25
        return pts

    @staticmethod
    def calc_flops(x):
        assert (x.shape[0] == 1)
        num_flops = x.numel() + 26 * x.shape[1]
        num_macs = 0
        return num_flops, num_macs
