"""
    Unclassified common routines for models in PyTorch.
"""

__all__ = ['Identity', 'BreakBlock', 'Flatten', 'SelectableDense', 'DenseBlock', 'NormActivation',
           'InterpolationBlock', 'ChannelShuffle', 'ChannelShuffle2', 'DucBlock', 'HeatmapMaxDetBlock']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Callable
from .activ import lambda_relu, create_activation_layer
from .norm import lambda_batchnorm1d, lambda_batchnorm2d, create_normalization_layer
from .conv import conv3x3_block


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


class Flatten(nn.Module):
    """
    Simple flatten module.
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


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
