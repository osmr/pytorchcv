"""
    Common routines for models in PyTorch.
"""

__all__ = ['round_channels', 'BreakBlock', 'SelectableDense', 'DenseBlock', 'NormActivation', 'InterpolationBlock',
           'ChannelShuffle', 'ChannelShuffle2', 'SEBlock', 'SABlock', 'SAConvBlock', 'saconv3x3_block', 'DucBlock',
           'Flatten', 'HeatmapMaxDetBlock']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Callable
from .activ import lambda_relu, lambda_sigmoid, create_activation_layer
from .norm import lambda_batchnorm1d, lambda_batchnorm2d, create_normalization_layer
from .conv import conv1x1, ConvBlock, conv3x3_block


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
