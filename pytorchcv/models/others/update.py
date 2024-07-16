import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from common import lambda_relu, conv1x1_block, conv3x3_block, conv7x7_block, ConvBlock, Concurrent


class FlowHead(nn.Module):
    def __init__(self,
                 input_dim=128,
                 hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class Conv1x1Branch(nn.Module):
    """
    Inception specific convolutional 1x1 branch block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    normalization : function or None
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 normalization: Callable[..., nn.Module | None] | None):
        super(Conv1x1Branch, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            normalization=normalization)

    def forward(self, x):
        x = self.conv(x)
        return x


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

    def forward(self, flow, corr):
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

    def forward(self, flow, corr):
        corr1 = self.conv_corr(corr)
        flow1 = self.conv_flow(flow)
        out = torch.cat([corr1, flow1], dim=1)
        out = self.conv_out(out)
        out = torch.cat([out, flow], dim=1)
        return out


class SmallUpdateBlock(nn.Module):
    def __init__(self,
                 corr_levels,
                 corr_radius,
                 hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(corr_levels=corr_levels, corr_radius=corr_radius)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self,
                 corr_levels,
                 corr_radius,
                 hidden_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_levels=corr_levels, corr_radius=corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
