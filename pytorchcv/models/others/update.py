import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from common import lambda_relu, lambda_sigmoid, lambda_tanh, conv1x1, conv3x3, conv1x1_block, conv3x3_block, ConvBlock


class ConvSeqBranch(nn.Module):
    """
    Inception specific convolutional sequence branch block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list(int) or tuple(int, ...)
        List of numbers of output channels.
    kernel_size_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of convolution window sizes.
    strides_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of strides of the convolution.
    padding_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of padding values for convolution layers.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or None, default lambda_relu()
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels_list: list[int] | tuple[int, ...],
                 kernel_size_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 strides_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 padding_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | None = lambda_relu()):
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
                padding=padding,
                bias=bias,
                normalization=normalization))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv_list(x)
        return x


class SmallMotionEncoder(nn.Module):
    def __init__(self,
                 corr_levels,
                 corr_radius):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2

        self.conv_corr = conv1x1_block(
            in_channels=cor_planes,
            out_channels=96,
            bias=True,
            normalization=None)
        self.conv_flow = ConvSeqBranch(
            in_channels=2,
            out_channels_list=(64, 32),
            kernel_size_list=(7, 3),
            strides_list=(1, 1),
            padding_list=(3, 1),
            bias=True,
            normalization=None)
        self.conv_out = conv3x3_block(
            in_channels=128,
            out_channels=80,
            bias=True,
            normalization=None)

    def forward(self, corr, flow):
        corr1 = self.conv_corr(corr)
        flow1 = self.conv_flow(flow)
        out = torch.cat([corr1, flow1], dim=1)
        out = self.conv_out(out)
        out = torch.cat([out, flow], dim=1)
        return out


class BasicMotionEncoder(nn.Module):
    def __init__(self,
                 corr_levels,
                 corr_radius):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2

        self.conv_corr = ConvSeqBranch(
            in_channels=cor_planes,
            out_channels_list=(256, 192),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            bias=True,
            normalization=None)
        self.conv_flow = ConvSeqBranch(
            in_channels=2,
            out_channels_list=(128, 64),
            kernel_size_list=(7, 3),
            strides_list=(1, 1),
            padding_list=(3, 1),
            bias=True,
            normalization=None)
        self.conv_out = conv3x3_block(
            in_channels=(64 + 192),
            out_channels=(128 - 2),
            bias=True,
            normalization=None)

    def forward(self, corr, flow):
        corr1 = self.conv_corr(corr)
        flow1 = self.conv_flow(flow)
        out = torch.cat([corr1, flow1], dim=1)
        out = self.conv_out(out)
        out = torch.cat([out, flow], dim=1)
        return out


class ConvGRU(nn.Module):
    """
    Convolutional GRU.

    Parameters
    ----------
    hidden_dim : int
        Hidden value size.
    input_dim : int
        Input value size.
    kernel_size : int or tuple(int, int), default 3
        Convolution window size.
    padding : int or tuple(int, int) or tuple(int, int, int, int), default 1
        Padding value for convolution layer.
    """
    def __init__(self,
                 hidden_dim,
                 input_dim,
                 kernel_size: int | tuple[int, int] = 3,
                 padding: int | tuple[int, int] | tuple[int, int, int, int] = 1):
        super(ConvGRU, self).__init__()
        sum_dim = hidden_dim + input_dim

        self.conv_z = ConvBlock(
            in_channels=sum_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
            normalization=None,
            activation=lambda_sigmoid())
        self.conv_r = ConvBlock(
            in_channels=sum_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
            normalization=None,
            activation=lambda_sigmoid())
        self.conv_q = ConvBlock(
            in_channels=sum_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
            normalization=None,
            activation=lambda_tanh())

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = self.conv_z(hx)

        r = self.conv_r(hx)
        q = torch.cat([r * h, x], dim=1)
        q = self.conv_q(q)

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self,
                 hidden_dim,
                 input_dim):
        super(SepConvGRU, self).__init__()
        self.horizontal_gru = ConvGRU(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            kernel_size=(1, 5),
            padding=(0, 2))
        self.vertical_gru = ConvGRU(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            kernel_size=(5, 1),
            padding=(2, 0))

    def forward(self, h, x):
        h = self.horizontal_gru(h, x)
        h = self.vertical_gru(h, x)
        return h


class FlowHead(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels):
        super(FlowHead, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            normalization=None)
        self.conv2 = conv3x3(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MaskHead(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels):
        super(MaskHead, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            normalization=None)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SmallUpdateBlock(nn.Module):
    def __init__(self,
                 corr_levels,
                 corr_radius,
                 hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(
            corr_levels=corr_levels,
            corr_radius=corr_radius)
        self.gru = ConvGRU(
            hidden_dim=hidden_dim,
            input_dim=(82 + 64))
        self.flow_head = FlowHead(
            in_channels=hidden_dim,
            mid_channels=128,
            out_channels=2)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(corr=corr, flow=flow)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(h=net, x=inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self,
                 corr_levels,
                 corr_radius,
                 hidden_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(
            corr_levels=corr_levels,
            corr_radius=corr_radius)
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim,
            input_dim=(128 + hidden_dim))
        self.flow_head = FlowHead(
            in_channels=hidden_dim,
            mid_channels=256,
            out_channels=2)
        self.mask = MaskHead(
            in_channels=128,
            mid_channels=256,
            out_channels=(64 * 9))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(corr=corr, flow=flow)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(h=net, x=inp)
        delta_flow = self.flow_head(net)

        # Scale mask to balence gradients:
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
