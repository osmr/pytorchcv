"""
    ProPainter (Recurrent Flow Completion), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['SecondOrderDeformableAlignment']

import math
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from typing import Callable
from common import (lambda_relu, create_activation_layer, lambda_leakyrelu, conv1x1, conv3x3, conv3x3_block,
                    InterpolationBlock)
from resnet import ResUnit


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
    def __init__(self,
                 channels: int):
        super(RFCBidirectionalPropagation, self).__init__()
        modules = ["backward_", "forward_"]
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channels = channels

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                x_in_channels=(2 * channels),
                cond_in_channels=(3 * channels),
                out_channels=channels,
                deform_groups=16,
                max_residue_magnitude=5)

            self.backbone[module] = nn.Sequential(
                conv3x3(
                    in_channels=((2 + i) * channels),
                    out_channels=channels,
                    bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                conv3x3(
                    in_channels=channels,
                    out_channels=channels,
                    bias=True),
            )

        self.fusion = conv1x1(
            in_channels=(2 * channels),
            out_channels=channels,
            bias=True)

    def forward(self, x):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats["spatial"] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ["backward_", "forward_"]:

            feats[module_name] = []

            frame_idx = range(0, t)
            mapping_idx = list(range(0, len(feats["spatial"])))
            mapping_idx += mapping_idx[::-1]

            if "backward" in module_name:
                frame_idx = frame_idx[::-1]

            feat_prop = x.new_zeros(b, self.channels, h, w)
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
        for i in range(0, t):
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


class DecoderUnit(nn.Module):
    """
    Decoder unit.

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
        super(DecoderUnit, self).__init__()
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


class RecurrentFlowCompleteNet(nn.Module):
    def __init__(self):
        super().__init__()
        man_activation = lambda_leakyrelu(negative_slope=0.2)

        self.downsample = ConvBlock3d(
            in_channels=3,
            out_channels=32,
            kernel_size=(1, 5, 5),
            stride=(1, 2, 2),
            padding=(0, 2, 2),
            padding_mode="replicate",
            activation=man_activation)

        self.encoder1 = nn.Sequential(
            P3dBlock(
                in_channels=32,
                out_channels=32,
                stride=1,
                activation=man_activation,
                final_activation=man_activation),
            P3dBlock(
                in_channels=32,
                out_channels=64,
                stride=2,
                activation=man_activation,
                final_activation=man_activation),
        )  # 4x

        self.encoder2 = nn.Sequential(
            P3dBlock(
                in_channels=64,
                out_channels=64,
                stride=1,
                activation=man_activation,
                final_activation=man_activation),
            P3dBlock(
                in_channels=64,
                out_channels=128,
                stride=2,
                activation=man_activation,
                final_activation=man_activation),
        )  # 8x

        self.mid_dilation = DilationBlock(
            channels=128,
            activation=man_activation)

        # feature propagation module
        self.feat_prop_module = RFCBidirectionalPropagation(128)

        self.decoder2 = DecoderUnit(
            in_channels=128,
            out_channels=64,
            activation=man_activation,
            final_activation=man_activation)

        self.decoder1 = DecoderUnit(
            in_channels=64,
            out_channels=32,
            activation=man_activation,
            final_activation=man_activation)

        self.upsample = DecoderUnit(
            in_channels=32,
            out_channels=2,
            activation=man_activation,
            final_activation=None)

        # edge loss
        edge_det_final_activation = lambda_leakyrelu(negative_slope=0.01)
        self.edgeDetector = EdgeDetection(
            in_channels=2,
            out_channels=1,
            mid_channels=16,
            activation=man_activation,
            final_activation=edge_det_final_activation)

    #     self._init_params()
    #
    # def _init_params(self):
    #     # Need to initial the weights of MSDeformAttn specifically
    #     for m in self.modules():
    #         if isinstance(m, SecondOrderDeformableAlignment):
    #             m.init_offset()

    def forward(self, masked_flows, masks):
        # masked_flows: b t-1 2 h w
        # masks: b t-1 2 h w
        b, t, _, h, w = masked_flows.size()
        masked_flows = masked_flows.permute(0, 2, 1, 3, 4)
        masks = masks.permute(0, 2, 1, 3, 4)

        x = torch.cat((masked_flows, masks), dim=1)

        x = self.downsample(x)

        feat_e1 = self.encoder1(x)

        feat_e2 = self.encoder2(feat_e1)  # b c t h w
        feat_mid = self.mid_dilation(feat_e2)  # b c t h w
        feat_mid = feat_mid.permute(0, 2, 1, 3, 4)  # b t c h w

        feat_prop = self.feat_prop_module(feat_mid)
        feat_prop = feat_prop.view(-1, 128, h // 8, w // 8)  # b*t c h w

        _, c, _, h_f, w_f = feat_e1.shape
        feat_e1 = feat_e1.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h_f, w_f)  # b*t c h w
        feat_d2 = self.decoder2(feat_prop) + feat_e1

        feat_d1 = self.decoder1(feat_d2)

        flow = self.upsample(feat_d1)
        if True:
        # if self.training:
            edge = self.edgeDetector(flow)
            edge = edge.view(b, t, 1, h, w)
        else:
            edge = None

        flow = flow.view(b, t, 2, h, w)

        return flow, edge

    def forward_bidirect_flow(self, masked_flows_bi, masks):
        """
        Args:
            masked_flows_bi: [masked_flows_f, masked_flows_b] | (b t-1 2 h w), (b t-1 2 h w)
            masks: b t 1 h w
        """
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        # mask flow
        masked_flows_forward = masked_flows_bi[0] * (1 - masks_forward)
        masked_flows_backward = masked_flows_bi[1] * (1 - masks_backward)

        # -- completion --
        # forward
        pred_flows_forward, pred_edges_forward = self.forward(masked_flows_forward, masks_forward)

        # backward
        masked_flows_backward = torch.flip(masked_flows_backward, dims=[1])
        masks_backward = torch.flip(masks_backward, dims=[1])
        pred_flows_backward, pred_edges_backward = self.forward(masked_flows_backward, masks_backward)
        pred_flows_backward = torch.flip(pred_flows_backward, dims=[1])
        if self.training:
            pred_edges_backward = torch.flip(pred_edges_backward, dims=[1])

        return [pred_flows_forward, pred_flows_backward], [pred_edges_forward, pred_edges_backward]

    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1 - masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1 - masks_backward)

        return pred_flows_forward, pred_flows_backward


def _test2():
    import re
    import os
    import numpy as np

    def convert_state_dict(src_checkpoint,
                           dst_checkpoint):

        src_param_keys = list(src_checkpoint.keys())

        upd_dict = {}

        list1 = list(filter(re.compile("edgeDetector.").search, src_param_keys))
        list1_u = [key.replace(".projection.0.", ".proj.conv.") for key in list1]
        list1_u = [key.replace(".mid_layer_1.0.", ".res_unit.body.conv1.conv.") for key in list1_u]
        list1_u = [key.replace(".mid_layer_2.0.", ".res_unit.body.conv2.conv.") for key in list1_u]
        list1_u = [key.replace(".out_layer.", ".out_conv.") for key in list1_u]
        for src_i, dst_i in zip(list1, list1_u):
            upd_dict[src_i] = dst_i

        list2 = list(filter(re.compile("feat_prop_module.deform_align.").search, src_param_keys))
        list2_u = [key.replace(".backward_.weight", ".backward_.deform_conv.weight") for key in list2]
        list2_u = [key.replace(".backward_.bias", ".backward_.deform_conv.bias") for key in list2_u]
        list2_u = [key.replace(".forward_.weight", ".forward_.deform_conv.weight") for key in list2_u]
        list2_u = [key.replace(".forward_.bias", ".forward_.deform_conv.bias") for key in list2_u]
        list2_u = [key.replace(".conv_offset.0.", ".conv_offset.conv1.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.2.", ".conv_offset.conv2.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.4.", ".conv_offset.conv3.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.6.", ".conv_offset.conv4.conv.") for key in list2_u]
        for src_i, dst_i in zip(list2, list2_u):
            upd_dict[src_i] = dst_i

        list3 = list(filter(re.compile("downsample.").search, src_param_keys))
        list3_u = [key.replace(".0.", ".conv.") for key in list3]
        for src_i, dst_i in zip(list3, list3_u):
            upd_dict[src_i] = dst_i

        list4 = list(filter(re.compile("encoder").search, src_param_keys))
        list4_u = [key.replace("encoder1.2.", "encoder1.1.") for key in list4]
        list4_u = [key.replace("encoder2.2.", "encoder2.1.") for key in list4_u]
        list4_u = [key.replace(".conv1.0.", ".conv1.conv.") for key in list4_u]
        list4_u = [key.replace(".conv2.0.", ".conv2.conv.") for key in list4_u]
        for src_i, dst_i in zip(list4, list4_u):
            upd_dict[src_i] = dst_i

        list5 = list(filter(re.compile("mid_dilation.").search, src_param_keys))
        list5_u = [key.replace(".0.", ".conv1.conv.") for key in list5]
        list5_u = [key.replace(".2.", ".conv2.conv.") for key in list5_u]
        list5_u = [key.replace(".4.", ".conv3.conv.") for key in list5_u]
        for src_i, dst_i in zip(list5, list5_u):
            upd_dict[src_i] = dst_i

        list6 = list(filter(re.compile("decoder").search, src_param_keys))
        list6_u = [key.replace(".0.", ".conv1.conv.") for key in list6]
        list6_u = [key.replace(".2.", ".conv2.") for key in list6_u]
        for src_i, dst_i in zip(list6, list6_u):
            upd_dict[src_i] = dst_i

        list7 = list(filter(re.compile("upsample.").search, src_param_keys))
        list7_u = [key.replace(".0.", ".conv1.conv.") for key in list7]
        list7_u = [key.replace(".2.", ".conv2.") for key in list7_u]
        for src_i, dst_i in zip(list7, list7_u):
            upd_dict[src_i] = dst_i

        list4_r = []

        for k, v in src_checkpoint.items():
            if k in upd_dict.keys():
                dst_checkpoint[upd_dict[k]] = src_checkpoint[k]
            else:
                if k not in list4_r:
                    dst_checkpoint[k] = src_checkpoint[k]
                else:
                    print("Remove: {}".format(k))
                    pass

    root_path = "../../../pytorchcv_data/test"
    rfc_model_file_name = "recurrent_flow_completion.pth"

    model_path = os.path.join(root_path, rfc_model_file_name)
    net_rfc = RecurrentFlowCompleteNet()

    src_checkpoint = torch.load(model_path, map_location="cpu")
    dst_checkpoint = net_rfc.state_dict()
    convert_state_dict(
        src_checkpoint,
        dst_checkpoint)
    net_rfc.load_state_dict(dst_checkpoint, strict=True)
    # ckpt = torch.load(model_path, map_location="cpu")
    # net_rfc.load_state_dict(ckpt, strict=True)

    for p in net_rfc.parameters():
        p.requires_grad = False
    net_rfc.eval()
    net_rfc = net_rfc.cuda()

    flow1_f_file_path = os.path.join(root_path, "gt_flow_f_00099.npy")
    flow2_f_file_path = os.path.join(root_path, "gt_flow_f_00100.npy")
    flow1_b_file_path = os.path.join(root_path, "gt_flow_b_00099.npy")
    flow2_b_file_path = os.path.join(root_path, "gt_flow_b_00100.npy")
    mask1_file_path = os.path.join(root_path, "flow_mask_00099.npy")
    mask2_file_path = os.path.join(root_path, "flow_mask_00100.npy")
    mask3_file_path = os.path.join(root_path, "flow_mask_00101.npy")
    flow_f_comp_file_path = os.path.join(root_path, "flow_f_comp.npy")
    flow_b_comp_file_path = os.path.join(root_path, "flow_b_comp.npy")
    flow1_f_np = np.load(flow1_f_file_path)
    flow2_f_np = np.load(flow2_f_file_path)
    flow1_b_np = np.load(flow1_b_file_path)
    flow2_b_np = np.load(flow2_b_file_path)
    mask1_np = np.load(mask1_file_path)
    mask2_np = np.load(mask2_file_path)
    mask3_np = np.load(mask3_file_path)
    flow_f_comp_np = np.load(flow_f_comp_file_path)
    flow_b_comp_np = np.load(flow_b_comp_file_path)

    flow_f_np = np.stack([flow1_f_np, flow2_f_np])[None]
    flow_b_np = np.stack([flow1_b_np, flow2_b_np])[None]
    mask_np = np.stack([mask1_np, mask2_np, mask3_np])[None]

    flow_f = torch.from_numpy(flow_f_np).cuda()
    flow_b = torch.from_numpy(flow_b_np).cuda()
    mask = torch.from_numpy(mask_np).cuda()

    (flow_f_comp, flow_b_comp), _ = net_rfc.forward_bidirect_flow(
        masked_flows_bi=(flow_f, flow_b),
        masks=mask)
    flow_f_comp, flow_b_comp = net_rfc.combine_flow(
        masked_flows_bi=(flow_f, flow_b),
        pred_flows_bi=(flow_f_comp, flow_b_comp),
        masks=mask)

    flow_f_comp_np_ = flow_f_comp[0].cpu().detach().numpy()
    flow_b_comp_np_ = flow_b_comp[0].cpu().detach().numpy()

    # np.save(os.path.join(root_path, "flow_f_comp.npy"), np.ascontiguousarray(flow_f_comp_np_))
    # np.save(os.path.join(root_path, "flow_b_comp.npy"), np.ascontiguousarray(flow_b_comp_np_))

    if not np.array_equal(flow_f_comp_np, flow_f_comp_np_):
        print("*")
    np.testing.assert_array_equal(flow_f_comp_np, flow_f_comp_np_)
    if not np.array_equal(flow_b_comp_np, flow_b_comp_np_):
        print("*")
    np.testing.assert_array_equal(flow_b_comp_np, flow_b_comp_np_)

    masks_forward = mask[:, :-1, ...].contiguous()
    masked_flows_forward = flow_f * (1 - masks_forward)
    pred_flows_f, pred_edges_f = net_rfc(masked_flows_forward, masks_forward)

    pred_flows_f_np_ = pred_flows_f[0].cpu().detach().numpy()
    pred_edges_f_np_ = pred_edges_f[0].cpu().detach().numpy()

    # np.save(os.path.join(root_path, "pred_flows_f.npy"), np.ascontiguousarray(pred_flows_f_np_))
    # np.save(os.path.join(root_path, "pred_edges_f.npy"), np.ascontiguousarray(pred_edges_f_np_))

    pred_flows_f_file_path = os.path.join(root_path, "pred_flows_f.npy")
    pred_edges_f_file_path = os.path.join(root_path, "pred_edges_f.npy")

    pred_flows_f_np = np.load(pred_flows_f_file_path)
    pred_edges_f_np = np.load(pred_edges_f_file_path)

    if not np.array_equal(pred_flows_f_np, pred_flows_f_np_):
        print("*")
    np.testing.assert_array_equal(pred_flows_f_np, pred_flows_f_np_)
    if not np.array_equal(pred_edges_f_np, pred_edges_f_np_):
        print("*")
    np.testing.assert_array_equal(pred_edges_f_np, pred_edges_f_np_)

    pass


# def _test():
#     import torch
#     from model_store import calc_net_weight_count
#
#     pretrained = False
#
#     models = [
#         raft_things,
#         raft_small,
#     ]
#
#     for model in models:
#
#         net = model(pretrained=pretrained)
#
#         # net.train()
#         net.eval()
#         weight_count = calc_net_weight_count(net)
#         print("m={}, {}".format(model.__name__, weight_count))
#         assert (model != raft_things or weight_count == 5257536)
#         assert (model != raft_small or weight_count == 990162)
#
#         batch = 4
#         height = 240
#         width = 432
#         x1 = torch.randn(batch, 3, height, width)
#         x2 = torch.randn(batch, 3, height, width)
#         y1, y2 = net(x1, x2)
#         # y1.sum().backward()
#         # y2.sum().backward()
#         assert (tuple(y1.size()) == (batch, 2, height // 8, width // 8))
#         assert (tuple(y2.size()) == (batch, 2, height, width))


if __name__ == "__main__":
    _test2()
