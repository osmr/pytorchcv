import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from common import (lambda_relu, lambda_sigmoid, lambda_tanh, lambda_batchnorm2d, lambda_instancenorm2d,
                    lambda_groupnorm, conv1x1, conv3x3, conv1x1_block, conv3x3_block, conv7x7_block, ConvBlock)


def bilinear_sampler(img,
                     coords,
                     mask=False):
    """
    Wrapper for grid_sample, uses pixel coordinates.
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


class CorrBlock:
    def __init__(self,
                 fmap1,
                 fmap2,
                 num_levels=4,
                 radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


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


class RAFT(nn.Module):
    def __init__(self,
                 small: bool,
                 dropout_rate: float = 0.0):
        super(RAFT, self).__init__()
        self.small = small
        self.dropout_rate = dropout_rate

        if self.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_levels = 4
            self.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4

        # feature network, context network, and update block
        if self.small:
            self.fnet = SmallEncoder(
                output_dim=128,
                norm_fn="instance",
                dropout_rate=self.dropout_rate)
            self.cnet = SmallEncoder(
                output_dim=hdim + cdim,
                norm_fn="none",
                dropout_rate=self.dropout_rate)
            self.update_block = SmallUpdateBlock(
                corr_levels=self.corr_levels,
                corr_radius=self.corr_radius,
                hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(
                output_dim=256,
                norm_fn="instance",
                dropout_rate=self.dropout_rate)
            self.cnet = BasicEncoder(
                output_dim=hdim + cdim,
                norm_fn="batch",
                dropout_rate=self.dropout_rate)
            self.update_block = BasicUpdateBlock(
                corr_levels=self.corr_levels,
                corr_radius=self.corr_radius,
                hidden_dim=hdim)

    @staticmethod
    def coords_grid(batch,
                    ht,
                    wd):
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    @staticmethod
    def initialize_flow(img):
        """
        Flow is represented as difference between two coordinate grids flow = coords1 - coords0.
        """
        N, C, H, W = img.shape
        coords0 = RAFT.coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = RAFT.coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    @staticmethod
    def upflow8(flow,
                mode="bilinear"):
        new_size = (8 * flow.shape[2], 8 * flow.shape[3])
        return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    @staticmethod
    def upsample_flow(flow, mask):
        """
        Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination.
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None):
        """
        Estimate optical flow between pair of frames.
        """
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = RAFT.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = RAFT.upflow8(coords1 - coords0)
            else:
                flow_up = RAFT.upsample_flow(coords1 - coords0, up_mask)

        return coords1 - coords0, flow_up
