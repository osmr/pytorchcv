"""
    RAFT, implemented in PyTorch.
    Original paper: 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,'
    https://arxiv.org/pdf/2003.12039.
"""

__all__ = ['RAFT', 'raft_things', 'raft_small']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from common import (lambda_relu, lambda_sigmoid, lambda_tanh, lambda_batchnorm2d, lambda_instancenorm2d, conv1x1,
                     conv3x3, conv3x3_block, conv7x7_block, ConvBlock)
from resnet import ResUnit
from inceptionv3 import ConvSeqBranch


def create_coords_grid(batch: int,
                       height: int,
                       width: int) -> torch.Tensor:
    """
    Create coordinate grid.

    Parameters
    ----------
    batch : int
        Batch size.
    height : int
        Height.
    width : int
        Width.

    Returns
    -------
    torch.Tensor
        Resulted tensor.
    """
    coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def initialize_flow(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create structures for flow as difference between two coordinate grids (flow = coords1 - coords0).

    Parameters
    ----------
    img : torch.Tensor
        Image for shape calculation.

    Returns
    -------
    torch.Tensor
        Coordinate grid for #0.
    torch.Tensor
        Coordinate grid for #1.
    """
    batch, _, height, width = img.size()
    coords0 = create_coords_grid(batch, height // 8, width // 8).to(img.device)
    coords1 = create_coords_grid(batch, height // 8, width // 8).to(img.device)
    return coords0, coords1


def upsample_flow_using_mask(flow: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
    """
    Upsample flow field [2, H/8, W/8] -> [2, H, W] using convex combination.

    Parameters
    ----------
    flow : torch.Tensor
        Flow.
    mask : torch.Tensor
        Mask.

    Returns
    -------
    torch.Tensor
        Upsampled flow.
    """
    batch, channels, height, width = flow.size()
    assert (channels == 2)

    mask = mask.view(batch, 1, 9, 8, 8, height, width)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(
        input=(8 * flow),
        kernel_size=[3, 3],
        padding=1)
    up_flow = up_flow.view(batch, 2, 9, 1, 1, height, width)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(batch, 2, 8 * height, 8 * width)


def upsample_flow_using_interpolation(flow: torch.Tensor,
                                      mode: str = "bilinear") -> torch.Tensor:
    """
    Upsample flow field [2, H/8, W/8] -> [2, H, W] using interpolation.

    Parameters
    ----------
    flow : torch.Tensor
        Flow.
    mode : str, default 'bilinear'
        Interpolation mode.

    Returns
    -------
    torch.Tensor
        Upsampled flow.
    """
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    up_flow = 8 * F.interpolate(
        input=flow,
        size=new_size,
        mode=mode,
        align_corners=True)
    return up_flow


def bilinear_sampler(img: torch.Tensor,
                     coords: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for grid_sample, uses pixel coordinates.

    Parameters
    ----------
    img : torch.Tensor
        Processed image.
    coords : torch.Tensor
        Coordinates.

    Returns
    -------
    torch.Tensor
        Sampled image.
    """
    height, width = img.shape[-2:]
    x_grid, y_grid = coords.split(split_size=[1, 1], dim=-1)
    x_grid = 2 * x_grid / (width - 1) - 1
    y_grid = 2 * y_grid / (height - 1) - 1

    grid = torch.cat([x_grid, y_grid], dim=-1)
    img = F.grid_sample(
        input=img,
        grid=grid,
        align_corners=True)
    return img


class CorrCalculator:
    """
    Correlation calculator.

    Parameters
    ----------
    fmap1 : torch.Tensor
        Feature map #1.
    fmap2 : torch.Tensor
        Feature map #2.
    radius : int
        Correlation radius.
    num_levels : int, default 4
        Number of correlation pyramid levels.
    """
    def __init__(self,
                 fmap1: torch.Tensor,
                 fmap2: torch.Tensor,
                 radius: int,
                 num_levels: int = 4):
        self.radius = radius
        self.num_levels = num_levels
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrCalculator.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    """
    Calculate correlation.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates.

    Returns
    -------
    torch.Tensor
        Correlation.
    """
    def __call__(self,
                 coords: torch.Tensor) -> torch.Tensor:
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
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    """
    Calculate correlation matrix for two feature maps.

    Parameters
    ----------
    fmap1 : torch.Tensor
        Feature map #1.
    fmap2 : torch.Tensor
        Feature map #2.

    Returns
    -------
    torch.Tensor
        Correlation matrix.
    """
    @staticmethod
    def corr(fmap1: torch.Tensor,
             fmap2: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = fmap1.shape
        fmap1 = fmap1.view(batch, channels, height * width)
        fmap2 = fmap2.view(batch, channels, height * width)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, height, width, 1, height, width)
        corr /= torch.sqrt(torch.tensor(channels).float())
        return corr


class RAFTEncoder(nn.Module):
    """
    RAFT feature/context encoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    init_block_channels : int
        Number of output channels for the initial unit.
    mid_channels : list(list(int))
        Number of output channels for each unit.
    final_block_channels : int
        Number of output channels for the final unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    normalization : function or None, default lambda_batchnorm2d()
        Lambda-function generator for normalization layer.
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    """
    def __init__(self,
                 in_channels: int,
                 init_block_channels: int,
                 mid_channels: list[list[int]],
                 final_block_channels: int,
                 bottleneck: bool,
                 normalization: Callable[..., nn.Module | None] | None = lambda_batchnorm2d(),
                 dropout_rate=0.0):
        super(RAFTEncoder, self).__init__()
        conv1_stride = False
        final_activation = lambda_relu()

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv7x7_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bias=True,
            normalization=normalization))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(mid_channels):
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


class RAFTMotionEncoder(nn.Module):
    """
    RAFT motion encoder.

    Parameters
    ----------
    corr_levels : int
        Correlation levels.
    corr_radius : int
        Correlation radius.
    corr_out_channels_list : tuple(int, ...)
        Numbers of output channels for correlation convolution.
    flow_out_channels_list : tuple(int, ...)
        Numbers of output channels for flow convolution.
    mout_in_channels : int
        Number of input channels for output convolution.
    mout_out_channels : int
        Number of output channels for output convolution.
    """
    def __init__(self,
                 corr_levels: int,
                 corr_radius: int,
                 corr_out_channels_list: tuple[int, ...],
                 flow_out_channels_list: tuple[int, ...],
                 mout_in_channels: int,
                 mout_out_channels: int):
        super(RAFTMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2

        if len(corr_out_channels_list) == 1:
            corr_kernel_size_list = (1,)
            corr_strides_list = (1,)
            corr_padding_list = (0,)
        else:
            assert (len(corr_out_channels_list) == 2)
            corr_kernel_size_list = (1, 3)
            corr_strides_list = (1, 1)
            corr_padding_list = (0, 1)

        self.conv_corr = ConvSeqBranch(
            in_channels=cor_planes,
            out_channels_list=corr_out_channels_list,
            kernel_size_list=corr_kernel_size_list,
            strides_list=corr_strides_list,
            padding_list=corr_padding_list,
            bias=True,
            normalization=None)
        self.conv_flow = ConvSeqBranch(
            in_channels=2,
            out_channels_list=flow_out_channels_list,
            kernel_size_list=(7, 3),
            strides_list=(1, 1),
            padding_list=(3, 1),
            bias=True,
            normalization=None)
        self.conv_out = conv3x3_block(
            in_channels=mout_in_channels,
            out_channels=mout_out_channels,
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
                 hidden_dim: int,
                 input_dim: int,
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
    """
    Separable convolutional GRU.

    Parameters
    ----------
    hidden_dim : int
        Hidden value size.
    input_dim : int
        Input value size.
    """
    def __init__(self,
                 hidden_dim: int,
                 input_dim: int):
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
    """
    Flow head block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int):
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
    """
    Mask head block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int):
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


class RAFTUpdateBlock(nn.Module):
    """
    RAFT udpate block.

    Parameters
    ----------
    corr_levels : int
        Correlation level.
    corr_radius : int
        Correlation radius.
    hidden_dim : int
        Hidden data size.
    corr_out_channels_list : tuple(int, ...)
        Number of output channels for the motion encoder correlation convolutions in the update block.
    flow_out_channels_list : tuple(int, ...)
        Number of output channels for the motion encoder flow convolutions in the update block.
    mout_in_channels : int
        Number of input channels for the motion encoder last convolution in the update block.
    mout_out_channels : int
        Number of output channels for the motion encoder last convolution in the update block.
    gru_class : type(nn.Module)
        GRU class.
    gru_input_dim : int
        Number of input channels for GRU in the update block.
    flow_mid_channels : int
        Number of middle channels for flow-head in the update block.
    mask_out_channels : int
        Number of output channels for mask in the update block.
    """
    def __init__(self,
                 corr_levels: int,
                 corr_radius: int,
                 hidden_dim: int,
                 corr_out_channels_list: tuple[int, ...],
                 flow_out_channels_list: tuple[int, ...],
                 mout_in_channels: int,
                 mout_out_channels: int,
                 gru_class: type[nn.Module],
                 gru_input_dim: int,
                 flow_mid_channels: int,
                 mask_out_channels: int):
        super(RAFTUpdateBlock, self).__init__()
        self.calc_mask = (mask_out_channels != 0)

        self.encoder = RAFTMotionEncoder(
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            corr_out_channels_list=corr_out_channels_list,
            flow_out_channels_list=flow_out_channels_list,
            mout_in_channels=mout_in_channels,
            mout_out_channels=mout_out_channels)
        self.gru = gru_class(
            hidden_dim=hidden_dim,
            input_dim=gru_input_dim)
        self.flow_head = FlowHead(
            in_channels=hidden_dim,
            mid_channels=flow_mid_channels,
            out_channels=2)
        if self.calc_mask:
            self.mask = MaskHead(
                in_channels=hidden_dim,
                mid_channels=flow_mid_channels,
                out_channels=mask_out_channels)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(corr=corr, flow=flow)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(h=net, x=inp)
        delta_flow = self.flow_head(net)

        if self.calc_mask:
            # Scale mask to balence gradients:
            mask = 0.25 * self.mask(net)
        else:
            mask = None

        return net, mask, delta_flow


class RAFT(nn.Module):
    """
    RAFT model from 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,' https://arxiv.org/pdf/2003.12039.

    Parameters
    ----------
    corr_levels : int
        Correlation level.
    corr_radius : int
        Correlation radius.
    hidden_dim : int
        Hidden data size.
    context_dim : int
        Context data size.
    encoder_init_block_channels : int
        Number of output channels for the initial unit in feature/context encoder networks.
    encoder_mid_channels : list(list(int))
        Number of output channels for each unit in feature/context encoder networks.
    fnet_final_block_channels : int
        Number of final block channels for the feature networks.
    encoder_bottleneck : bool
        Whether to use bottleneck in feature/context encoder networks.
    cnet_normalize : bool
        Whether to normalize the context network.
    corr_out_channels_list : tuple(int, ...)
        Number of output channels for the motion encoder correlation convolutions in the update block.
    flow_out_channels_list : tuple(int, ...)
        Number of output channels for the motion encoder flow convolutions in the update block.
    mout_in_channels : int
        Number of input channels for the motion encoder last convolution in the update block.
    mout_out_channels : int
        Number of output channels for the motion encoder last convolution in the update block.
    gru_class : type(nn.Module)
        GRU class.
    gru_input_dim : int
        Number of input channels for GRU in the update block.
    flow_mid_channels : int
        Number of middle channels for flow-head in the update block.
    mask_out_channels : int
        Number of output channels for mask in the update block.
    in_normalize : bool, default False
        Whether to normalize input images.
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input channels.
    """
    def __init__(self,
                 corr_levels: int,
                 corr_radius: int,
                 hidden_dim: int,
                 context_dim: int,
                 encoder_init_block_channels: int,
                 encoder_mid_channels: list[list[int]],
                 fnet_final_block_channels: int,
                 encoder_bottleneck: bool,
                 cnet_normalize: bool,
                 corr_out_channels_list: tuple[int, ...],
                 flow_out_channels_list: tuple[int, ...],
                 mout_in_channels: int,
                 mout_out_channels: int,
                 gru_class: type[nn.Module],
                 gru_input_dim: int,
                 flow_mid_channels: int,
                 mask_out_channels: int,
                 in_normalize: bool = False,
                 dropout_rate: float = 0.0,
                 in_channels: int = 3):
        super(RAFT, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_radius = corr_radius
        self.in_normalize = in_normalize
        fnet_normalization = lambda_instancenorm2d()
        cnet_normalization = lambda_batchnorm2d() if cnet_normalize else None

        # feature network
        self.fnet = RAFTEncoder(
            in_channels=in_channels,
            init_block_channels=encoder_init_block_channels,
            mid_channels=encoder_mid_channels,
            final_block_channels=fnet_final_block_channels,
            bottleneck=encoder_bottleneck,
            normalization=fnet_normalization,
            dropout_rate=dropout_rate)

        # context network
        self.cnet = RAFTEncoder(
            in_channels=in_channels,
            init_block_channels=encoder_init_block_channels,
            mid_channels=encoder_mid_channels,
            final_block_channels=hidden_dim + context_dim,
            bottleneck=encoder_bottleneck,
            normalization=cnet_normalization,
            dropout_rate=dropout_rate)

        # update block
        self.update_block = RAFTUpdateBlock(
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_dim=hidden_dim,
            corr_out_channels_list=corr_out_channels_list,
            flow_out_channels_list=flow_out_channels_list,
            mout_in_channels=mout_in_channels,
            mout_out_channels=mout_out_channels,
            gru_class=gru_class,
            gru_input_dim=gru_input_dim,
            flow_mid_channels=flow_mid_channels,
            mask_out_channels=mask_out_channels)

    def forward(self, image1, image2, iters=12, flow_init=None):
        """
        Estimate optical flow between pair of frames.
        """
        if self.in_normalize:
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_calc = CorrCalculator(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_calc(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upsample_flow_using_interpolation(coords1 - coords0)
            else:
                flow_up = upsample_flow_using_mask(coords1 - coords0, up_mask)

        return coords1 - coords0, flow_up


def get_raft(version: str,
             model_name: str | None = None,
             pretrained: bool = False,
             root: str = os.path.join("~", ".torch", "models"),
             **kwargs) -> nn.Module:
    """
    Create RAFT model with specific parameters.

    Parameters
    ----------
    version : str
        Version of RAFT ('basic' or 'small').
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
    if version == "basic":
        corr_levels = 4
        corr_radius = 4
        hidden_dim = 128
        context_dim = 128
        encoder_init_block_channels = 64
        encoder_mid_channels = [[64, 64], [96, 96], [128, 128]]
        fnet_final_block_channels = 256
        encoder_bottleneck = False
        cnet_normalize = True

        corr_out_channels_list = (256, 192)
        flow_out_channels_list = (128, 64)
        mout_in_channels = (64 + 192)
        mout_out_channels = (128 - 2)

        gru_class = SepConvGRU
        gru_input_dim = 128 + 128
        flow_mid_channels = 256
        mask_out_channels = 64 * 9
    elif version == "small":
        corr_levels = 4
        corr_radius = 3
        hidden_dim = 96
        context_dim = 64
        encoder_init_block_channels = 32
        encoder_mid_channels = [[32, 32], [64, 64], [96, 96]]
        fnet_final_block_channels = 128
        encoder_bottleneck = True
        cnet_normalize = False

        corr_out_channels_list = (96,)
        flow_out_channels_list = (64, 32)
        mout_in_channels = 128
        mout_out_channels = 80

        gru_class = ConvGRU
        gru_input_dim = 82 + 64
        flow_mid_channels = 128
        mask_out_channels = 0
    else:
        raise ValueError("Unsupported RAFT version {}".format(version))

    net = RAFT(
        corr_levels=corr_levels,
        corr_radius=corr_radius,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        encoder_init_block_channels=encoder_init_block_channels,
        encoder_mid_channels=encoder_mid_channels,
        fnet_final_block_channels=fnet_final_block_channels,
        encoder_bottleneck=encoder_bottleneck,
        cnet_normalize=cnet_normalize,
        corr_out_channels_list=corr_out_channels_list,
        flow_out_channels_list=flow_out_channels_list,
        mout_in_channels=mout_in_channels,
        mout_out_channels=mout_out_channels,
        gru_class=gru_class,
        gru_input_dim=gru_input_dim,
        flow_mid_channels=flow_mid_channels,
        mask_out_channels=mask_out_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def raft_things(**kwargs) -> nn.Module:
    """
    RAFT-Things model from 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,'
    https://arxiv.org/pdf/2003.12039.

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
    return get_raft(
        version="basic",
        model_name="raft_things",
        **kwargs)


def raft_small(**kwargs) -> nn.Module:
    """
    RAFT-Small model from 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,'
    https://arxiv.org/pdf/2003.12039.

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
    return get_raft(
        version="small",
        model_name="raft_small",
        **kwargs)


def _test2():
    import re
    import numpy as np

    def convert_state_dict(src_checkpoint,
                           dst_checkpoint,
                           small=False):

        src_param_keys = list(src_checkpoint.keys())

        upd_dict = {}

        list1 = list(filter(re.compile("net.conv1.").search, src_param_keys))
        list1_u = [key.replace("net.conv1.", "net.features.init_block.conv.") for key in list1]
        for src_i, dst_i in zip(list1, list1_u):
            upd_dict[src_i] = dst_i

        list2 = list(filter(re.compile("net.conv2.").search, src_param_keys))
        list2_u = [key.replace("net.conv2.", "net.features.final_block.") for key in list2]
        for src_i, dst_i in zip(list2, list2_u):
            upd_dict[src_i] = dst_i

        list3 = list(filter(re.compile("net.norm1.").search, src_param_keys))
        list3_u = [key.replace("net.norm1.", "net.features.init_block.bn.") for key in list3]
        for src_i, dst_i in zip(list3, list3_u):
            upd_dict[src_i] = dst_i

        list5 = list(filter(re.compile("update_block.encoder.").search, src_param_keys))
        list5_u = [key.replace(".conv.", ".conv_out.conv.") for key in list5]
        if small:
            # list5_u = [key.replace(".convc1.", ".conv_corr.conv.") for key in list5_u]
            list5_u = [key.replace(".convc1.", ".conv_corr.conv_list.conv1.conv.") for key in list5_u]
            list5_u = [key.replace(".convc2.", ".conv_corr.conv_list.conv2.conv.") for key in list5_u]
        else:
            list5_u = [key.replace(".convc1.", ".conv_corr.conv_list.conv1.conv.") for key in list5_u]
            list5_u = [key.replace(".convc2.", ".conv_corr.conv_list.conv2.conv.") for key in list5_u]
        list5_u = [key.replace(".convf1.", ".conv_flow.conv_list.conv1.conv.") for key in list5_u]
        list5_u = [key.replace(".convf2.", ".conv_flow.conv_list.conv2.conv.") for key in list5_u]
        for src_i, dst_i in zip(list5, list5_u):
            upd_dict[src_i] = dst_i

        list6 = list(filter(re.compile("update_block.gru.").search, src_param_keys))
        if small:
            list6_u = [key.replace(".convz.", ".conv_z.conv.") for key in list6]
            list6_u = [key.replace(".convr.", ".conv_r.conv.") for key in list6_u]
            list6_u = [key.replace(".convq.", ".conv_q.conv.") for key in list6_u]
        else:
            list6_u = [key.replace(".convz1.", ".horizontal_gru.conv_z.conv.") for key in list6]
            list6_u = [key.replace(".convr1.", ".horizontal_gru.conv_r.conv.") for key in list6_u]
            list6_u = [key.replace(".convq1.", ".horizontal_gru.conv_q.conv.") for key in list6_u]
            list6_u = [key.replace(".convz2.", ".vertical_gru.conv_z.conv.") for key in list6_u]
            list6_u = [key.replace(".convr2.", ".vertical_gru.conv_r.conv.") for key in list6_u]
            list6_u = [key.replace(".convq2.", ".vertical_gru.conv_q.conv.") for key in list6_u]
        for src_i, dst_i in zip(list6, list6_u):
            upd_dict[src_i] = dst_i

        list7 = list(filter(re.compile("update_block.flow_head.").search, src_param_keys))
        list7_u = [key.replace(".conv1.", ".conv1.conv.") for key in list7]
        for src_i, dst_i in zip(list7, list7_u):
            upd_dict[src_i] = dst_i

        list8 = list(filter(re.compile("update_block.mask.").search, src_param_keys))
        list8_u = [key.replace(".0.", ".conv1.conv.") for key in list8]
        list8_u = [key.replace(".2.", ".conv2.") for key in list8_u]
        for src_i, dst_i in zip(list8, list8_u):
            upd_dict[src_i] = dst_i

        list4 = list(filter(re.compile("net.layer").search, src_param_keys))
        list4_u = [key.replace("net.layer1.0.", "net.features.stage1.unit1.body.") for key in list4]
        list4_u = [key.replace("net.layer1.1.", "net.features.stage1.unit2.body.") for key in list4_u]
        list4_u = [key.replace("net.layer2.0.", "net.features.stage2.unit1.body.") for key in list4_u]
        list4_u = [key.replace("net.layer2.1.", "net.features.stage2.unit2.body.") for key in list4_u]
        list4_u = [key.replace("net.layer3.0.", "net.features.stage3.unit1.body.") for key in list4_u]
        list4_u = [key.replace("net.layer3.1.", "net.features.stage3.unit2.body.") for key in list4_u]
        list4_u = [key.replace(".body.downsample.0.", ".identity_conv.conv.") for key in list4_u]
        list4_u = [key.replace(".body.downsample.1.", ".identity_conv.bn.") for key in list4_u]
        list4_u = [key.replace(".conv1.", ".conv1.conv.") for key in list4_u]
        list4_u = [key.replace(".conv2.", ".conv2.conv.") for key in list4_u]
        if small:
            list4_u = [key.replace(".conv3.", ".conv3.conv.") for key in list4_u]
        list4_u = [key.replace(".norm1.", ".conv1.bn.") for key in list4_u]
        list4_u = [key.replace(".norm2.", ".conv2.bn.") for key in list4_u]

        if small:
            list4_r = list(filter(re.compile(".norm4.").search, list4))
        else:
            list4_r = list(filter(re.compile(".norm3.").search, list4))

        for src_i, dst_i in zip(list4, list4_u):
            if src_i not in list4_r:
                upd_dict[src_i] = dst_i

        for k, v in src_checkpoint.items():
            if k in upd_dict.keys():
                dst_checkpoint[upd_dict[k]] = src_checkpoint[k]
            else:
                if k not in list4_r:
                    dst_checkpoint[k] = src_checkpoint[k]
                else:
                    print("Remove: {}".format(k))
                    pass

    def initialize_RAFT(model_path='weights/raft-things.pth',
                        device='cuda',
                        small=False):
        """Initializes the RAFT model.
        """
        if small:
            net = raft_small()
        else:
            net = raft_things()

        src_checkpoint = torch.load(model_path, map_location="cpu")

        # net_tmp = torch.nn.DataParallel(net)
        # net_tmp.load_state_dict(checkpoint)
        # checkpoint = net_tmp.module.cpu().state_dict()

        # dst_checkpoint = net.state_dict()
        # convert_state_dict(
        #     src_checkpoint,
        #     dst_checkpoint,
        #     small)

        dst_checkpoint = src_checkpoint

        net.load_state_dict(dst_checkpoint)

        # torch.save(net.state_dict(), model_path[:-4] + "2" + model_path[-4:])

        net.to(device)

        return net

    class RAFT_bi(nn.Module):
        """Flow completion loss"""
        def __init__(self,
                     model_path='weights/raft-things.pth',
                     device='cuda',
                     small=False):
            super().__init__()
            self.fix_raft = initialize_RAFT(
                model_path,
                device=device,
                small=small)

            for p in self.fix_raft.parameters():
                p.requires_grad = False

            self.eval()

        def forward(self, gt_local_frames, iters=20):
            b, l_t, c, h, w = gt_local_frames.size()
            # print(gt_local_frames.shape)

            with torch.no_grad():
                gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
                gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)
                # print(gtlf_1.shape)

                _, gt_flows_forward = self.fix_raft(gtlf_1, gtlf_2, iters=iters)
                _, gt_flows_backward = self.fix_raft(gtlf_2, gtlf_1, iters=iters)

            gt_flows_forward = gt_flows_forward.view(b, l_t - 1, 2, h, w)
            gt_flows_backward = gt_flows_backward.view(b, l_t - 1, 2, h, w)

            return gt_flows_forward, gt_flows_backward

    def sub_test(raft_small: bool = False):
        raft_iter = 20
        root_path = "../../../pytorchcv_data/test"

        if raft_small:
            raft_model_file_name = "raft-small_2.pth"
            y1_file_name = "y1_s.npy"
            y2_file_name = "y2_s.npy"
        else:
            raft_model_file_name = "raft-things_2.pth"
            y1_file_name = "y1.npy"
            y2_file_name = "y2.npy"

        fix_raft = RAFT_bi(
            model_path=os.path.join(root_path, raft_model_file_name),
            device="cuda",
            small=raft_small)

        x_file_path = os.path.join(root_path, "x.npy")
        y1_file_path = os.path.join(root_path, y1_file_name)
        y2_file_path = os.path.join(root_path, y2_file_name)
        x = np.load(x_file_path)
        y1 = np.load(y1_file_path)
        y2 = np.load(y2_file_path)

        frames = torch.from_numpy(x).cuda()

        flows_f, flows_b = fix_raft(
            gt_local_frames=frames,
            iters=raft_iter)

        y1_ = flows_f.cpu().detach().numpy()
        y2_ = flows_b.cpu().detach().numpy()

        # np.save(os.path.join(root_path, "y1_s.npy"), np.ascontiguousarray(y1_))
        # np.save(os.path.join(root_path, "y2_s.npy"), np.ascontiguousarray(y2_))

        if not np.array_equal(y1, y1_):
            print("*")
        np.testing.assert_array_equal(y1, y1_)
        if not np.array_equal(y2, y2_):
            print("*")
        np.testing.assert_array_equal(y2, y2_)

        pass

    sub_test(raft_small=True)
    sub_test(raft_small=False)


def _test():
    import torch
    from model_store import calc_net_weight_count

    pretrained = False

    models = [
        raft_things,
        raft_small,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != raft_things or weight_count == 5257536)
        assert (model != raft_small or weight_count == 990162)

        batch = 4
        height = 240
        width = 432
        x1 = torch.randn(batch, 3, height, width)
        x2 = torch.randn(batch, 3, height, width)
        y1, y2 = net(x1, x2)
        # y1.sum().backward()
        # y2.sum().backward()
        assert (tuple(y1.size()) == (batch, 2, height // 8, width // 8))
        assert (tuple(y2.size()) == (batch, 2, height, width))


if __name__ == "__main__":
    _test2()
