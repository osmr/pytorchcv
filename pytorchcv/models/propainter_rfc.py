"""
    ProPainter (Recurrent Flow Completion), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['PPRecurrentFlowComplete', 'propainter_rfc', 'SecondOrderDeformableAlignment',
           'calc_bidirectional_opt_flow_completion_by_pprfc']

import os
import math
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from typing import Callable
from .common.activ import lambda_relu, lambda_leakyrelu, create_activation_layer
from .common.conv import conv1x1, conv3x3_block
from .common.arch import Hourglass
from .common.tutti import InterpolationBlock, BreakBlock
from .resnet import ResUnit, ResBlock


class SecondOrderDeformableAlignment(nn.Module):
    """
    Second-order deformable alignment module.

    Parameters
    ----------
    x_in_channels : int
        Number of `x` input channels.
    cond_in_channels : int
        Number of `cond` input channels.
    out_channels : int
        Number of output channels.
    deform_groups : int
        Number of deformable groups.
    max_residue_magnitude : int
        Maximal residue magnitude.
    """
    def __init__(self,
                 x_in_channels: int,
                 cond_in_channels: int,
                 out_channels: int,
                 deform_groups: int,
                 max_residue_magnitude: int):
        super(SecondOrderDeformableAlignment, self).__init__()
        self.max_residue_magnitude = max_residue_magnitude

        cond_channels = [out_channels, out_channels, out_channels, 27 * deform_groups]
        cond_activation = lambda_leakyrelu(negative_slope=0.1)

        self.conv_offset = nn.Sequential()
        for i, cond_out_channels in enumerate(cond_channels):
            cond_activation_i = cond_activation if (i != len(cond_channels) - 1) else None
            self.conv_offset.add_module("conv{}".format(i + 1), conv3x3_block(
                in_channels=cond_in_channels,
                out_channels=cond_out_channels,
                bias=True,
                normalization=None,
                activation=cond_activation_i))
            cond_in_channels = cond_out_channels

        self.deform_conv = DeformConv2d(
            in_channels=x_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self._init_params()

    def _init_params(self):
        n = self.deform_conv.in_channels
        for k in self.deform_conv.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.deform_conv.weight.data.uniform_(-stdv, stdv)
        if self.deform_conv.bias is not None:
            self.deform_conv.bias.data.zero_()

        nn.init.constant_(self.conv_offset[-1].conv.weight, 0.0)
        nn.init.constant_(self.conv_offset[-1].conv.bias, 0.0)

    def forward(self, x, cond, flow=None):
        y = self.conv_offset(cond)
        offset1, offset2, mask = torch.chunk(y, 3, dim=1)

        offset = torch.cat((offset1, offset2), dim=1)
        offset = self.max_residue_magnitude * torch.tanh(offset)
        if flow is not None:
            offset += flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        mask = torch.sigmoid(mask)

        out = self.deform_conv(
            input=x,
            offset=offset,
            mask=mask)
        return out


class RFCBidirectionalPropagation(nn.Module):
    """
    Bidirectional propagation module (specific for Recurrent Flow Completion task).

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    """
    def __init__(self,
                 channels: int):
        super(RFCBidirectionalPropagation, self).__init__()
        self.channels = channels

        activation = lambda_leakyrelu(negative_slope=0.1)
        modules = ["backward_", "forward_"]

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                x_in_channels=(2 * channels),
                cond_in_channels=(3 * channels),
                out_channels=channels,
                deform_groups=16,
                max_residue_magnitude=5)
            self.backbone[module] = ResBlock(
                in_channels=((2 + i) * channels),
                out_channels=channels,
                stride=1,
                bias=True,
                normalization=None,
                activation=activation)

        self.fusion = conv1x1(
            in_channels=(2 * channels),
            out_channels=channels,
            bias=True)

    def forward(self, x):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        batch, time, channels, height, width = x.shape
        assert (channels == self.channels)

        feats = {}
        feats["spatial"] = [x[:, i, :, :, :] for i in range(0, time)]

        for module_name in ["backward_", "forward_"]:

            feats[module_name] = []

            frame_idx = range(0, time)
            mapping_idx = list(range(0, len(feats["spatial"])))
            mapping_idx += mapping_idx[::-1]

            if "backward" in module_name:
                frame_idx = frame_idx[::-1]

            feat_prop = x.new_zeros(batch, channels, height, width)
            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][mapping_idx[idx]]
                if i > 0:
                    cond_n1 = feat_prop

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feat_prop)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:  # second-order features
                        feat_n2 = feats[module_name][-2]
                        cond_n2 = feat_n2

                    # condition information, cond(flow warped 1st/2nd feature):
                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)  # two order feat_prop -1 & -2
                    feat_prop = self.deform_align[module_name](x=feat_prop, cond=cond)

                # fuse current features
                feat = ([feat_current] + [feats[k][idx] for k in feats if k not in ["spatial", module_name]] +
                        [feat_prop])

                feat = torch.cat(feat, dim=1)
                # embed current features
                feat_prop = feat_prop + self.backbone[module_name](feat)

                feats[module_name].append(feat_prop)

            # end for
            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, time):
            align_feats = [feats[k].pop(0) for k in feats if k != "spatial"]
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x


class ConvBlock3d(nn.Module):
    """
    Standard 3D convolution block with activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int, int)
        Convolution window size.
    stride : int or tuple(int, int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int, int), default 0
        Padding value for convolution layer.
    dilation : int or tuple(int, int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default True
        Whether the layer uses a bias vector.
    padding_mode : str, default 'zeros'
        Padding mode.
    activation : function or None, default lambda_relu()
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int, int],
                 stride: int | tuple[int, int, int] = 1,
                 padding: int | tuple[int, int, int] = 0,
                 dilation: int | tuple[int, int, int] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 activation: Callable[..., nn.Module | None] | None = lambda_relu()):
        super(ConvBlock3d, self).__init__()
        self.activate = (activation is not None)

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)
        if self.activate:
            self.activ = create_activation_layer(activation)
            if self.activ is None:
                self.activate = False

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x3x3_block(padding: int | tuple[int, int, int] = (0, 1, 1),
                    **kwargs) -> nn.Module:
    """
    1x3x3 version of the standard 3D convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int, int), default (0, 1, 1)
        Padding value for convolution layer.
    dilation : int or tuple(int, int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default True
        Whether the layer uses a bias vector.
    padding_mode : str, default 'zeros'
        Padding mode.
    activation : function or None, default lambda_relu()
        Lambda-function generator for activation layer.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return ConvBlock3d(
        kernel_size=(1, 3, 3),
        padding=padding,
        **kwargs)


class P3dBlock(nn.Module):
    """
    Simple ResNet-like block with 3D convolutions and with specific kernel sizes.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int
        Strides of the convolution for (height x width).
    bias : bool, default True
        Whether the layer uses a bias vector.
    activation : function, default lambda_relu()
        Lambda-function generator for activation layer in the main convolution block.
    final_activation : function or None, default None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 bias: bool = True,
                 activation: Callable[..., nn.Module] = lambda_relu(),
                 final_activation: Callable[..., nn.Module | None] | None = None):
        super(P3dBlock, self).__init__()
        self.conv1 = conv1x3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, stride, stride),
            bias=bias,
            activation=activation)
        self.conv2 = ConvBlock3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(2, 0, 0),
            dilation=(2, 1, 1),
            bias=bias,
            activation=final_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DilationBlock(nn.Module):
    """
    Dilation block.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    activation : function
        Lambda-function generator for activation layer in the convolution blocks.
    """
    def __init__(self,
                 channels: int,
                 activation: Callable[..., nn.Module]):
        super(DilationBlock, self).__init__()
        self.conv1 = conv1x3x3_block(
            in_channels=channels,
            out_channels=channels,
            padding=(0, 3, 3),
            dilation=(1, 3, 3),
            activation=activation)  # p = d*(k-1)/2
        self.conv2 = conv1x3x3_block(
            in_channels=channels,
            out_channels=channels,
            padding=(0, 2, 2),
            dilation=(1, 2, 2),
            activation=activation)
        self.conv3 = conv1x3x3_block(
            in_channels=channels,
            out_channels=channels,
            padding=(0, 1, 1),
            dilation=(1, 1, 1),
            activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class RFCDecoderUnit(nn.Module):
    """
    Decoder unit (specific for Recurrent Flow Completion task).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function
        Lambda-function generator for activation layer in the main convolution block.
    final_activation : function or None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable[..., nn.Module],
                 final_activation: Callable[..., nn.Module | None] | None):
        super(RFCDecoderUnit, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            bias=True,
            normalization=None,
            activation=activation)
        self.up = InterpolationBlock(scale_factor=2)
        self.conv2 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            normalization=None,
            activation=final_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        return x


class EdgeDetection(nn.Module):
    """
    Edge detection block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    activation : function
        Lambda-function generator for activation layer in the main convolution block.
    final_activation : function or None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 activation: Callable[..., nn.Module],
                 final_activation: Callable[..., nn.Module | None] | None):
        super(EdgeDetection, self).__init__()
        self.proj = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            normalization=None,
            activation=activation)
        self.res_unit = ResUnit(
            in_channels=mid_channels,
            out_channels=mid_channels,
            bias=True,
            normalization=None,
            bottleneck=False,
            activation=activation,
            final_activation=final_activation)
        self.out_conv = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.res_unit(x)
        x = self.out_conv(x)
        x = torch.sigmoid(x)
        return x


class EncoderUnit(nn.Module):
    """
    RFC specific encoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function
        Lambda-function generator for activation layer in convolution blocks.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable[..., nn.Module]):
        super(EncoderUnit, self).__init__()
        self.block1 = P3dBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            activation=activation,
            final_activation=activation)
        self.block2 = P3dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            activation=activation,
            final_activation=activation)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class InitBlock(nn.Module):
    """
    RFC specific initial block.

    Input tensor size should be (batch, time, channels, height, width).
    Output tensor size will be (batch, channels, time, height, width).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function
        Lambda-function generator for activation layer in a convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable[..., nn.Module]):
        super(InitBlock, self).__init__()
        self.conv = ConvBlock3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 5, 5),
            stride=(1, 2, 2),
            padding=(0, 2, 2),
            padding_mode="replicate",
            activation=activation)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        return x


class MainUnit(nn.Module):
    """
    RFC specific main unit.

    Input tensor size should be (batch, channels, time, height, width).
    Output tensor size will be (batch * time, channels, height, width).

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    activation : function
        Lambda-function generator for activation layer in a convolution block.
    """
    def __init__(self,
                 channels: int,
                 activation: Callable[..., nn.Module]):
        super(MainUnit, self).__init__()
        self.mid_dilation = DilationBlock(
            channels=channels,
            activation=activation)
        self.feat_prop_module = RFCBidirectionalPropagation(channels)

    def forward(self, x):
        x = self.mid_dilation(x)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.feat_prop_module(x)

        batch, time, channels, height, width = x.size()
        x = x.view(batch * time, channels, height, width)
        return x


class ReshapeBlock(nn.Module):
    """
    RFC specific reshape skip block.

    Input tensor size should be (batch, channels, time, height, width).
    Output tensor size will be (batch * time, channels, height, width).
    """
    def __init__(self):
        super(ReshapeBlock, self).__init__()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        batch, time, channels, height, width = x.size()
        x = x.view(batch * time, channels, height, width)
        return x


class PPRecurrentFlowComplete(nn.Module):
    """
    Recurrent Flow Completion part of ProPainter model from 'ProPainter: Improving Propagation and Transformer for Video
    Inpainting,' https://arxiv.org/pdf/2309.03897.

    Parameters
    ----------
    calc_edge : bool, default False
        Whether to calculate edge image.
    in_channels : int, default 3
        Number of input channels.
    """
    def __init__(self,
                 calc_edge: bool = False,
                 in_channels: int = 3):
        super(PPRecurrentFlowComplete, self).__init__()
        self.calc_edge = calc_edge
        man_activation = lambda_leakyrelu(negative_slope=0.2)

        down_seq = nn.Sequential()
        down_seq.add_module("down1", InitBlock(
            in_channels=in_channels,
            out_channels=32,
            activation=man_activation))
        down_seq.add_module("down2", EncoderUnit(
            in_channels=32,
            out_channels=64,
            activation=man_activation))
        down_seq.add_module("down3", EncoderUnit(
            in_channels=64,
            out_channels=128,
            activation=man_activation))

        up_seq = nn.Sequential()
        up_seq.add_module("up1", RFCDecoderUnit(
            in_channels=32,
            out_channels=2,
            activation=man_activation,
            final_activation=None))
        up_seq.add_module("up2", RFCDecoderUnit(
            in_channels=64,
            out_channels=32,
            activation=man_activation,
            final_activation=man_activation))
        up_seq.add_module("up3", RFCDecoderUnit(
            in_channels=128,
            out_channels=64,
            activation=man_activation,
            final_activation=man_activation))

        skip_seq = nn.Sequential()
        skip_seq.add_module("skip1", BreakBlock())
        skip_seq.add_module("skip2", BreakBlock())
        skip_seq.add_module("skip3", ReshapeBlock())
        skip_seq.add_module("skip4", MainUnit(
            channels=128,
            activation=man_activation))

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            merge_type="add")

        edge_det_final_activation = lambda_leakyrelu(negative_slope=0.01)
        self.edgeDetector = EdgeDetection(
            in_channels=2,
            out_channels=1,
            mid_channels=16,
            activation=man_activation,
            final_activation=edge_det_final_activation)

    def forward(self, masked_flows, masks):
        batch, time, channels, height, width = masked_flows.size()
        assert (channels == 2)
        assert (height % 8 == 0)
        assert (width % 8 == 0)
        assert (masked_flows.shape[0] == masks.shape[0])
        assert (masked_flows.shape[1] == masks.shape[1])
        assert (masks.shape[2] == 1)
        assert (masked_flows.shape[3] == masks.shape[3])
        assert (masked_flows.shape[4] == masks.shape[4])

        x = torch.cat((masked_flows, masks), dim=2)

        flow = self.hg(x)

        # if self.training:
        if self.calc_edge:
            edge = self.edgeDetector(flow)
            edge = edge.view(batch, time, 1, height, width)
        else:
            edge = None

        flow = flow.view(batch, time, 2, height, width)

        return flow, edge


def get_propainter_rfc(model_name: str | None = None,
                       pretrained: bool = False,
                       root: str = os.path.join("~", ".torch", "models"),
                       **kwargs) -> nn.Module:
    """
    Create Recurrent Flow Completion part of ProPainter model with specific parameters.

    Parameters
    ----------
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
    net = PPRecurrentFlowComplete(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .common.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def propainter_rfc(**kwargs) -> nn.Module:
    """
    ProPainter-RFC model from 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.

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
    return get_propainter_rfc(
        model_name="propainter_rfc",
        **kwargs)


def calc_bidirectional_opt_flow_completion_by_pprfc(net: PPRecurrentFlowComplete,
                                                    flows: torch.Tensor,
                                                    flow_masks: torch.Tensor,
                                                    combine_flows: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complete bidirectional optical flow (based on ProPainter's Recurrent Flow Completion net) on video.
    Batch dimension for frames is interpreted as time.

    Parameters
    ----------
    net: PPRecurrentFlowComplete
        Optical flow completion model.
    flows : torch.Tensor
        Bidirectional flows with size: (time, channels=4, height, width).
    flow_masks : torch.Tensor
        Flow masks with size: (time, channels=2, height, width).
    combine_flows : bool, default True
        Whether to combine predicted flows with original flows.

    Returns
    -------
    torch.Tensor
        Forward/Backward completed flow with size: (time, channels=4, height, width).
    torch.Tensor
        Forward/Backward edge images with size: (time, channels=2, height, width).
    """
    assert (len(flows.shape) == 4)
    assert (len(flows.shape) == len(flow_masks.shape))
    assert (flows.shape[0] == flow_masks.shape[0])
    assert (flows.shape[1] == 4)
    assert (flow_masks.shape[1] == 2)
    assert (flows.shape[2] == flow_masks.shape[2])
    assert (flows.shape[3] == flow_masks.shape[3])

    flows_forward, flows_backward = torch.split(flows, [2, 2], dim=1)
    masks_forward, masks_backward = torch.split(flow_masks, [1, 1], dim=1)

    masked_flows_forward = flows_forward * (1 - masks_forward)
    masked_flows_backward = flows_backward * (1 - masks_backward)

    comp_flows_forward, edges_forward = net.forward(
        masked_flows=masked_flows_forward[None],
        masks=masks_forward[None])
    comp_flows_forward = comp_flows_forward[0]
    if edges_forward is not None:
        edges_forward = edges_forward[0]

    masked_flows_backward_inv = torch.flip(masked_flows_backward, dims=[0])
    masks_backward_inv = torch.flip(masks_backward, dims=[0])
    comp_flows_backward_inv, edges_backward_inv = net.forward(
        masked_flows=masked_flows_backward_inv[None],
        masks=masks_backward_inv[None])
    comp_flows_backward_inv = comp_flows_backward_inv[0]
    if edges_backward_inv is not None:
        edges_backward_inv = edges_backward_inv[0]
    comp_flows_backward = torch.flip(comp_flows_backward_inv, dims=[0])
    if edges_backward_inv is not None:
        edges_backward = torch.flip(edges_backward_inv, dims=[0])
    else:
        edges_backward = None

    if combine_flows:
        comp_flows_forward = comp_flows_forward * masks_forward + masked_flows_forward
        comp_flows_backward = comp_flows_backward * masks_backward + masked_flows_backward

    comp_flows = torch.cat((comp_flows_forward, comp_flows_backward), dim=1)

    if edges_forward is not None:
        edges = torch.cat((edges_forward, edges_backward), dim=1)
    else:
        edges = None

    assert (comp_flows.shape == flows.shape)

    return comp_flows, edges


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        propainter_rfc,
    ]

    for model in models:

        net = model(pretrained=pretrained, calc_edge=True)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != propainter_rfc or weight_count == 5079555)

        batch = 4
        time = 5
        height = 240
        width = 432
        masked_flows = torch.randn(batch, time, 2, height, width)
        masks = torch.randn(batch, time, 1, height, width)
        flow, edge = net(masked_flows, masks)
        # flow.sum().backward()
        # edge.sum().backward()
        assert (tuple(flow.size()) == (batch, time, 2, height, width))
        assert (tuple(edge.size()) == (batch, time, 1, height, width))


if __name__ == "__main__":
    _test()
