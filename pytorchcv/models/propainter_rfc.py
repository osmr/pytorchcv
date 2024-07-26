"""
    ProPainter (Recurrent Flow Completion), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single
import torchvision
from torchvision.ops import DeformConv2d
from common import lambda_leakyrelu, conv1x1, conv3x3, conv3x3_block
from resnet import ResUnit


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# class ModulatedDeformConv2d(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  deform_groups=1,
#                  bias=True):
#         super(ModulatedDeformConv2d, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.deform_groups = deform_groups
#         self.with_bias = bias
#         # enable compatibility with nn.Conv2d
#         self.transposed = False
#         self.output_padding = _single(0)
#
#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.init_weights()
#
#     def init_weights(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1.0 / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.zero_()
#
#         if hasattr(self, "conv_offset"):
#             self.conv_offset.weight.data.zero_()
#             self.conv_offset.bias.data.zero_()
#
#     def forward(self, x, offset, mask):
#         pass
#
#
# class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
#     """Second-order deformable alignment module."""
#     def __init__(self, *args, **kwargs):
#         self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 5)
#
#         super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)
#
#         self.conv_offset = nn.Sequential(
#             conv3x3(
#                 in_channels=(3 * self.out_channels),
#                 out_channels=self.out_channels,
#                 bias=True),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             conv3x3(
#                 in_channels=self.out_channels,
#                 out_channels=self.out_channels,
#                 bias=True),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             conv3x3(
#                 in_channels=self.out_channels,
#                 out_channels=self.out_channels,
#                 bias=True),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             conv3x3(
#                 in_channels=self.out_channels,
#                 out_channels=(27 * self.deform_groups),
#                 bias=True),
#         )
#         self.init_offset()
#
#     def init_offset(self):
#         constant_init(self.conv_offset[-1], val=0, bias=0)
#
#     def forward(self, x, extra_feat):
#         out = self.conv_offset(extra_feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#
#         # offset
#         offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
#         offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
#         offset = torch.cat([offset_1, offset_2], dim=1)
#
#         # mask
#         mask = torch.sigmoid(mask)
#
#         return torchvision.ops.deform_conv2d(
#             x,
#             offset,
#             self.weight,
#             self.bias,
#             self.stride,
#             self.padding,
#             self.dilation,
#             mask)


class SecondOrderDeformableAlignment(nn.Module):
    """Second-order deformable alignment module."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 deform_groups: int = 16):
        super(SecondOrderDeformableAlignment, self).__init__()
        self.max_residue_magnitude = 5

        self.conv_offset = nn.Sequential(
            conv3x3(
                in_channels=(3 * out_channels),
                out_channels=out_channels,
                bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            conv3x3(
                in_channels=out_channels,
                out_channels=out_channels,
                bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            conv3x3(
                in_channels=out_channels,
                out_channels=out_channels,
                bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            conv3x3(
                in_channels=out_channels,
                out_channels=(27 * deform_groups),
                bias=True),
        )

        self.deform_conv = DeformConv2d(
            in_channels=in_channels,
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

        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        out = self.deform_conv(
            input=x,
            offset=offset,
            mask=mask)
        return out

        # return torchvision.ops.deform_conv2d(
        #     x,
        #     offset,
        #     self.weight,
        #     self.bias,
        #     self.stride,
        #     self.padding,
        #     self.dilation,
        #     mask)


class BidirectionalPropagation(nn.Module):
    def __init__(self,
                 channels):
        super(BidirectionalPropagation, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channels = channels

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                in_channels=(2 * channels),
                out_channels=channels,
                deform_groups=16)

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
        feats['spatial'] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ['backward_', 'forward_']:

            feats[module_name] = []

            frame_idx = range(0, t)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]

            feat_prop = x.new_zeros(b, self.channels, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]
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
                    feat_prop = self.deform_align[module_name](feat_prop, cond)

                # fuse current features
                feat = ([feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] +
                        [feat_prop])

                feat = torch.cat(feat, dim=1)
                # embed current features
                feat_prop = feat_prop + self.backbone[module_name](feat)

                feats[module_name].append(feat_prop)

            # end for
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x


class deconv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=True)
        return self.conv(x)


class P3DBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=bias),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(2, 0, 0),
                dilation=(2, 1, 1),
                bias=bias))

    def forward(self, feats):
        feat1 = self.conv1(feats)
        feat2 = self.conv2(feat1)
        output = feat2
        return output


class EdgeDetection(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int):
        super(EdgeDetection, self).__init__()
        main_activation = lambda_leakyrelu(negative_slope=0.2)
        final_activation = lambda_leakyrelu(negative_slope=0.01)

        self.proj = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=True,
            normalization=None,
            activation=main_activation)

        self.res_unit = ResUnit(
            in_channels=mid_channels,
            out_channels=mid_channels,
            bias=True,
            normalization=None,
            bottleneck=False,
            activation=main_activation,
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
        self.downsample = nn.Sequential(
            nn.Conv3d(
                3,
                32,
                kernel_size=(1, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
                padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True))

        self.encoder1 = nn.Sequential(
            P3DBlock(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            P3DBlock(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 4x

        self.encoder2 = nn.Sequential(
            P3DBlock(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            P3DBlock(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 8x

        self.mid_dilation = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 3, 3),
                dilation=(1, 3, 3)),  # p = d*(k-1)/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 2, 2),
                dilation=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                dilation=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # feature propagation module
        self.feat_prop_module = BidirectionalPropagation(128)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(128, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 4x

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 32),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 2x

        self.upsample = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(32, 2)
        )

        # edge loss
        self.edgeDetector = EdgeDetection(
            in_channels=2,
            out_channels=1,
            mid_channels=16)

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
        for src_i, dst_i in zip(list2, list2_u):
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

    flow_f_file_path = os.path.join(root_path, "gt_flow_f_00100.npy")
    flow_b_file_path = os.path.join(root_path, "gt_flow_b_00100.npy")
    mask1_file_path = os.path.join(root_path, "flow_mask_00100.npy")
    mask2_file_path = os.path.join(root_path, "flow_mask_00101.npy")
    # flow_f_comp_file_path = os.path.join(root_path, "pred_flow_f_00100.npy")
    # flow_b_comp_file_path = os.path.join(root_path, "pred_flow_b_00100.npy")
    flow_f_comp_file_path = os.path.join(root_path, "flow_f_comp.npy")
    flow_b_comp_file_path = os.path.join(root_path, "flow_b_comp.npy")
    flow_f_np = np.load(flow_f_file_path)
    flow_b_np = np.load(flow_b_file_path)
    mask1_np = np.load(mask1_file_path)
    mask2_np = np.load(mask2_file_path)
    flow_f_comp_np = np.load(flow_f_comp_file_path)
    flow_b_comp_np = np.load(flow_b_comp_file_path)

    flow_f_np = flow_f_np[None, None]
    flow_b_np = flow_b_np[None, None]
    mask_np = np.stack([mask1_np, mask2_np])[None]

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

    flow_f_comp_np_ = flow_f_comp[0, 0].cpu().detach().numpy()
    flow_b_comp_np_ = flow_b_comp[0, 0].cpu().detach().numpy()

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

    pred_flows_f_np_ = pred_flows_f[0, 0].cpu().detach().numpy()
    pred_edges_f_np_ = pred_edges_f[0, 0].cpu().detach().numpy()

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
