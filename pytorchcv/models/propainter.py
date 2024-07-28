"""
    ProPainter (Recurrent Flow Completion), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

import math
import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single
from torchvision.ops import DeformConv2d, deform_conv2d
from einops import rearrange
from typing import Callable
from common import (lambda_relu, create_activation_layer, lambda_leakyrelu, conv1x1, conv3x3, conv3x3_block,
                    InterpolationBlock)
from resnet import ResUnit
from propainter_rfc import SecondOrderDeformableAlignment


def flow_warp(x,
              flow,
              interpolation="bilinear",
              padding_mode="zeros",
              align_corners=True):
    """
    Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw,
                       flow_bw,
                       alpha1=0.01,
                       alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw


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
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.zero_()
#
#         if hasattr(self, 'conv_offset'):
#             self.conv_offset.weight.data.zero_()
#             self.conv_offset.bias.data.zero_()
#
#     def forward(self, x, offset, mask):
#         pass
#
#
# def constant_init(module, val, bias=0):
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.constant_(module.weight, val)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)
#
#
# class DeformableAlignment(nn.Module):
#     """
#     Second-order deformable alignment module.
#     """
#     def __init__(self,
#                  x_in_channels: int,
#                  cond_in_channels: int,
#                  out_channels: int,
#                  deform_groups: int,
#                  max_residue_magnitude: int):
#         super(DeformableAlignment, self).__init__()
#         self.max_residue_magnitude = max_residue_magnitude
#
#         self.conv_offset = nn.Sequential(
#             nn.Conv2d(
#                 2 * out_channels + 2 + 1 + 2,
#                 out_channels, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),
#         )
#
#         self.deform_conv = DeformConv2d(
#             in_channels=x_in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             padding=1)
#
#         self.init_offset()
#
#     def init_offset(self):
#         constant_init(self.conv_offset[-1], val=0, bias=0)
#
#     def forward(self, x, cond, flow):
#         out = self.conv_offset(cond)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#
#         # offset
#         offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
#         offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
#
#         # mask
#         mask = torch.sigmoid(mask)
#
#         out = self.deform_conv(
#             input=x,
#             offset=offset,
#             mask=mask)
#         return out
#
#         # out = deform_conv2d(
#         #     x,
#         #     offset,
#         #     self.weight,
#         #     self.bias,
#         #     self.stride,
#         #     self.padding,
#         #     self.dilation,
#         #     mask)
#         # return out


class BidirectionalPropagation(nn.Module):
    def __init__(self,
                 channel,
                 learnable=True):
        super(BidirectionalPropagation, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ["backward_1", "forward_1"]
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    x_in_channels=channel,
                    cond_in_channels=(2 * channel + 2 + 1 + 2),
                    out_channels=channel,
                    deform_groups=16,
                    max_residue_magnitude=3)
                    # channel,
                    # channel,
                    # 3,
                    # padding=1,
                    # deform_groups=16,
                    # max_residue_magnitude=3)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

    def binary_mask(self,
                    mask,
                    th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        # return mask.float()
        return mask.to(mask)

    def forward(self,
                x,
                flows_forward,
                flows_backward,
                mask,
                interpolation="bilinear"):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        feats, masks = {}, {}
        feats["input"] = [x[:, i, :, :, :] for i in range(0, t)]
        masks["input"] = [mask[:, i, :, :, :] for i in range(0, t)]

        prop_list = ["backward_1", "forward_1"]
        cache_list = ["input"] + prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if "backward" in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask, mask_current], dim=1)
                        feat_prop = self.deform_align[module_name](x=feat_prop, cond=cond, flow=flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                        mask_prop_valid = self.binary_mask(mask_prop_valid)

                        union_vaild_mask = self.binary_mask(mask_current * flow_vaild_mask * (1 - mask_prop_valid))
                        feat_prop = union_vaild_mask * feat_warped + (1 - union_vaild_mask) * feat_current
                        # update mask
                        mask_prop = self.binary_mask(mask_current * (1 - (flow_vaild_mask * (1 - mask_prop_valid))))

                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(feats["backward_1"], dim=1).view(-1, c, h, w)
        outputs_f = torch.stack(feats["forward_1"], dim=1).view(-1, c, h, w)

        if self.learnable:
            mask_in = mask.view(-1, 2, h, w)
            masks_b, masks_f = None, None
            outputs = self.fuse(torch.cat([outputs_b, outputs_f, mask_in], dim=1)) + x.view(-1, c, h, w)
        else:
            # masks_b = torch.stack(masks['backward_1'], dim=1)
            masks_f = torch.stack(masks["forward_1"], dim=1)
            outputs = outputs_f

        return outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w), \
            outputs.view(b, -1, c, h, w), masks_f


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        batch, _, x_height, x_width = x.size()
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, height, width = x0.size()
                assert (height == x_height // 4)
                assert (width == x_width // 4)
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                y = x0.view(batch, g, -1, height, width)
                o = out.view(batch, g, -1, height, width)
                out = torch.cat([y, o], 2).view(batch, -1, height, width)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class SoftSplit(nn.Module):
    def __init__(self,
                 channel,
                 hidden,
                 kernel_size,
                 stride,
                 padding):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self,
                 channel,
                 hidden,
                 kernel_size,
                 stride,
                 padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat


def window_partition(x,
                     window_size,
                     n_head):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, n_head, T, window_size, window_size, C//n_head)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1], window_size[1], n_head, C // n_head)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows


class SparseWindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 n_head,
                 window_size,
                 pool_size=(4, 4),
                 qkv_bias=True,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 pooling_token=True):
        super().__init__()
        assert dim % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=ks, stride=stride, padding=(0, 0), groups=dim)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        # self.expand_size = tuple(i // 2 for i in window_size)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))

    def forward(self, x, mask=None, T_ind=None, attn_mask=None):
        b, t, h, w, c = x.shape  # 20 36
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.n_head
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0]  # 20
        new_w = n_ww * self.window_size[1]  # 36
        pad_r = new_w - w
        pad_b = new_h - h
        # reverse order
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0)
            mask = F.pad(mask, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        win_q = window_partition(q.contiguous(), self.window_size, self.n_head).view(b, n_wh * n_ww, self.n_head, t,
                                                                                     w_h * w_w, c_head)
        win_k = window_partition(k.contiguous(), self.window_size, self.n_head).view(b, n_wh * n_ww, self.n_head, t,
                                                                                     w_h * w_w, c_head)
        win_v = window_partition(v.contiguous(), self.window_size, self.n_head).view(b, n_wh * n_ww, self.n_head, t,
                                                                                     w_h * w_w, c_head)
        # roll_k and roll_v
        if any(i > 0 for i in self.expand_size):
            (k_tl, v_tl) = map(
                lambda a: torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3)),
                               (k, v))
            (k_bl, v_bl) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3)),
                               (k, v))
            (k_br, v_br) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3)),
                               (k, v))

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda a: window_partition(a, self.window_size, self.n_head).view(b, n_wh * n_ww, self.n_head, t,
                                                                                  w_h * w_w, c_head),
                (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda a: window_partition(a, self.window_size, self.n_head).view(b, n_wh * n_ww, self.n_head, t,
                                                                                  w_h * w_w, c_head),
                (v_tl, v_tr, v_bl, v_br))
            rool_k = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 4).contiguous()
            rool_v = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows),
                               4).contiguous()  # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            # mask out tokens in current window
            rool_k = rool_k[:, :, :, :, self.valid_ind_rolled]
            rool_v = rool_v[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_k.shape[4]
            rool_k = rool_k.view(b, n_wh * n_ww, self.n_head, t, roll_N, c // self.n_head)
            rool_v = rool_v.view(b, n_wh * n_ww, self.n_head, t, roll_N, c // self.n_head)
            win_k = torch.cat((win_k, rool_k), dim=4)
            win_v = torch.cat((win_v, rool_v), dim=4)
        else:
            win_k = win_k
            win_v = win_v

        # pool_k and pool_v
        if self.pooling_token:
            pool_x = self.pool_layer(x.view(b * t, new_h, new_w, c).permute(0, 3, 1, 2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0, 2, 3, 1).view(b, t, p_h, p_w, c)
            # pool_k
            pool_k = self.key(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)  # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_k = pool_k.view(b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head).permute(0, 1, 5, 2, 3, 4, 6)
            pool_k = pool_k.contiguous().view(b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head)
            win_k = torch.cat((win_k, pool_k), dim=4)
            # pool_v
            pool_v = self.value(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1,
                                                            1)  # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_v = pool_v.view(b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head).permute(0, 1, 5, 2, 3, 4, 6)
            pool_v = pool_v.contiguous().view(b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head)
            win_v = torch.cat((win_v, pool_v), dim=4)

        # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
        out = torch.zeros_like(win_q)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(b * l_t, new_h, new_w))
        mask = mask.view(b, l_t, n_wh * n_ww)
        mask = torch.sum(mask, dim=1)  # [b, n_wh*n_ww]
        for i in range(win_q.shape[0]):
            # For masked windows:
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(mask_n, self.n_head, t * w_h * w_w, c_head)
                win_k_t = win_k[i, mask_ind_i]
                win_v_t = win_v[i, mask_ind_i]
                # mask out key and value
                if T_ind is not None:
                    # key [n_wh*n_ww, n_head, t, w_h*w_w, c_head]
                    win_k_t = win_k_t[:, :, T_ind.view(-1)].view(mask_n, self.n_head, -1, c_head)
                    # value
                    win_v_t = win_v_t[:, :, T_ind.view(-1)].view(mask_n, self.n_head, -1, c_head)
                else:
                    win_k_t = win_k_t.view(n_wh * n_ww, self.n_head, t * w_h * w_w, c_head)
                    win_v_t = win_v_t.view(n_wh * n_ww, self.n_head, t * w_h * w_w, c_head)

                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                att_t = F.softmax(att_t, dim=-1)
                att_t = self.attn_drop(att_t)
                y_t = att_t @ win_v_t

                out[i, mask_ind_i] = y_t.view(-1, self.n_head, t, w_h * w_w, c_head)

            # For unmasked windows:
            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            win_q_s = win_q[i, unmask_ind_i]
            win_k_s = win_k[i, unmask_ind_i, :, :, :w_h * w_w]
            win_v_s = win_v[i, unmask_ind_i, :, :, :w_h * w_w]

            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            y_s = att_s @ win_v_s
            out[i, unmask_ind_i] = y_s

        # re-assemble all head outputs side by side
        out = out.view(b, n_wh, n_ww, self.n_head, t, w_h, w_w, c_head)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)

        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]

        # output projection
        out = self.proj_drop(self.proj(out))
        return out


class FusionFeedForward(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim=1960,
                 t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        assert t2t_params is not None
        self.t2t_params = t2t_params
        self.kernel_shape = reduce((lambda x, y: x * y), t2t_params['kernel_size'])  # 49

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        x = self.fc2(x)
        return x


class TemporalSparseTransformer(nn.Module):
    def __init__(self,
                 dim,
                 n_head,
                 window_size,
                 pool_size,
                 norm_layer=nn.LayerNorm,
                 t2t_params=None):
        super().__init__()
        self.window_size = window_size
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, t2t_params=t2t_params)

    def forward(self,
                x,
                fold_x_size,
                mask=None,
                T_ind=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        B, T, H, W, C = x.shape  # 20 36

        shortcut = x
        x = self.norm1(x)
        att_x = self.attention(x, mask, T_ind)

        # FFN
        x = shortcut + att_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), fold_x_size).view(B, T, H, W, C)

        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_head,
                 window_size,
                 pool_size,
                 depths,
                 t2t_params=None):
        super().__init__()
        blocks = []
        for i in range(depths):
            blocks.append(TemporalSparseTransformer(
                dim,
                n_head,
                window_size,
                pool_size,
                t2t_params=t2t_params))
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths

    def forward(self,
                x,
                fold_x_size,
                l_mask=None,
                t_dilation=2):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            l_mask: local mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        assert self.depths % t_dilation == 0, "wrong t_dilation input."
        T = x.size(1)
        T_ind = [torch.arange(i, T, t_dilation) for i in range(t_dilation)] * (self.depths // t_dilation)

        for i in range(0, self.depths):
            x = self.transformer[i](x, fold_x_size, l_mask, T_ind[i])

        return x


class InpaintGenerator(nn.Module):
    def __init__(self):
        super(InpaintGenerator, self).__init__()
        channel = 128
        hidden = 512

        # encoder
        self.encoder = Encoder()

        # decoder
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        # feature propagation module
        self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True)

        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlock(
            dim=hidden,
            n_head=num_heads,
            window_size=window_size,
            pool_size=pool_size,
            depths=depths,
            t2t_params=t2t_params)

        self._init_weights()

    def _init_weights(self,
                      init_type="normal",
                      gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("InstanceNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def img_propagation(self,
                        masked_frames,
                        completed_flows,
                        masks,
                        interpolation="nearest"):
        _, _, prop_frames, updated_masks = self.img_prop_module(
            x=masked_frames,
            flows_forward=completed_flows[0],
            flows_backward=completed_flows[1],
            mask=masks,
            interpolation=interpolation)
        return prop_frames, updated_masks

    def forward(self,
                masked_frames,
                completed_flows,
                masks_in,
                masks_updated,
                num_local_frames,
                interpolation="bilinear",
                t_dilation=2):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """
        l_t = num_local_frames
        batch, time, _, ori_height, ori_width = masked_frames.size()

        # Extracting features:
        enc_feat = self.encoder(torch.cat([
            masked_frames.view(batch * time, 3, ori_height, ori_width),
            masks_in.view(batch * time, 1, ori_height, ori_width),
            masks_updated.view(batch * time, 1, ori_height, ori_width)], dim=1))
        _, channels, height, width = enc_feat.size()
        local_feat = enc_feat.view(batch, time, channels, height, width)[:, :l_t, ...]
        ref_feat = enc_feat.view(batch, time, channels, height, width)[:, l_t:, ...]
        fold_feat_size = (height, width)

        ds_flows_f = F.interpolate(
            input=completed_flows[0].view(-1, 2, ori_height, ori_width),
            scale_factor=(1 / 4),
            mode="bilinear",
            align_corners=False).view(batch, l_t - 1, 2, height, width) / 4.0
        ds_flows_b = F.interpolate(
            completed_flows[1].view(-1, 2, ori_height, ori_width),
            scale_factor=(1 / 4),
            mode="bilinear",
            align_corners=False).view(batch, l_t - 1, 2, height, width) / 4.0
        ds_mask_in = F.interpolate(
            masks_in.reshape(-1, 1, ori_height, ori_width),
            scale_factor=(1 / 4),
            mode="nearest").view(batch, time, 1, height, width)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local = F.interpolate(
            masks_updated[:, :l_t].reshape(-1, 1, ori_height, ori_width),
            scale_factor=(1 / 4),
            mode="nearest").view(batch, l_t, 1, height, width)

        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, height, width))
            mask_pool_l = mask_pool_l.view(batch, time, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.view(-1, 1, height, width))
            mask_pool_l = mask_pool_l.view(batch, l_t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))

        prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
        _, _, local_feat, _ = self.feat_prop_module(
            x=local_feat,
            flows_forward=ds_flows_f,
            flows_backward=ds_flows_b,
            mask=prop_mask_in,
            interpolation=interpolation)
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        trans_feat = self.ss(enc_feat.view(-1, channels, height, width), batch, fold_feat_size)
        mask_pool_l = rearrange(mask_pool_l, "b t c h w -> b t h w c").contiguous()
        trans_feat = self.transformers(
            x=trans_feat,
            fold_x_size=fold_feat_size,
            l_mask=mask_pool_l,
            t_dilation=t_dilation)
        trans_feat = self.sc(trans_feat, time, fold_feat_size)
        trans_feat = trans_feat.view(batch, time, -1, height, width)

        enc_feat = enc_feat + trans_feat

        if self.training:
            output = self.decoder(enc_feat.view(-1, channels, height, width))
            output = torch.tanh(output).view(batch, time, 3, ori_height, ori_width)
        else:
            output = self.decoder(enc_feat[:, :l_t].view(-1, channels, height, width))
            output = torch.tanh(output).view(batch, l_t, 3, ori_height, ori_width)

        return output


def _test2():
    import re
    import os
    import numpy as np

    def convert_state_dict(src_checkpoint,
                           dst_checkpoint):

        src_param_keys = list(src_checkpoint.keys())

        upd_dict = {}

        list2 = list(filter(re.compile("feat_prop_module.deform_align.").search, src_param_keys))
        list2_u = [key.replace(".backward_1.weight", ".backward_1.deform_conv.weight") for key in list2]
        list2_u = [key.replace(".backward_1.bias", ".backward_1.deform_conv.bias") for key in list2_u]
        list2_u = [key.replace(".forward_1.weight", ".forward_1.deform_conv.weight") for key in list2_u]
        list2_u = [key.replace(".forward_1.bias", ".forward_1.deform_conv.bias") for key in list2_u]
        list2_u = [key.replace(".conv_offset.0.", ".conv_offset.conv1.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.2.", ".conv_offset.conv2.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.4.", ".conv_offset.conv3.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.6.", ".conv_offset.conv4.conv.") for key in list2_u]
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

    root_path = "../../../pytorchcv_data/test_a"
    pp_model_file_name = "ProPainter.pth"

    model_path = os.path.join(root_path, pp_model_file_name)
    net_pp = InpaintGenerator()

    src_checkpoint = torch.load(model_path, map_location="cpu")
    dst_checkpoint = net_pp.state_dict()
    convert_state_dict(
        src_checkpoint,
        dst_checkpoint)
    net_pp.load_state_dict(dst_checkpoint, strict=True)
    # ckpt = torch.load(model_path, map_location="cpu")
    # net_pp.load_state_dict(ckpt, strict=True)

    for p in net_pp.parameters():
        p.requires_grad = False
    net_pp.eval()
    net_pp = net_pp.cuda()

    frame1_file_path = os.path.join(root_path, "frame_00100.npy")
    frame2_file_path = os.path.join(root_path, "frame_00101.npy")
    frame1_np = np.load(frame1_file_path)
    frame2_np = np.load(frame2_file_path)

    mask_dilated1_file_path = os.path.join(root_path, "mask_dilated_00100.npy")
    mask_dilated2_file_path = os.path.join(root_path, "mask_dilated_00101.npy")
    mask_dilated1_np = np.load(mask_dilated1_file_path)
    mask_dilated2_np = np.load(mask_dilated2_file_path)

    pred_flow_f_file_path = os.path.join(root_path, "pred_flow_f_00100.npy")
    pred_flow_b_file_path = os.path.join(root_path, "pred_flow_b_00100.npy")
    pred_flow_f_np = np.load(pred_flow_f_file_path)
    pred_flow_b_np = np.load(pred_flow_b_file_path)

    frames_np = np.stack([frame1_np, frame2_np])[None]
    frames = torch.from_numpy(frames_np).cuda()
    masks_dilated_np = np.stack([mask_dilated1_np, mask_dilated2_np])[None]
    masks_dilated = torch.from_numpy(masks_dilated_np).cuda()
    masked_frames = frames * (1 - masks_dilated)

    pred_flow_f_np = pred_flow_f_np[None, None]
    pred_flow_b_np = pred_flow_b_np[None, None]
    pred_flow_f = torch.from_numpy(pred_flow_f_np).cuda()
    pred_flow_b = torch.from_numpy(pred_flow_b_np).cuda()

    prop_imgs, updated_local_masks = net_pp.img_propagation(
        masked_frames=masked_frames,
        completed_flows=(pred_flow_f, pred_flow_b),
        masks=masks_dilated,
        interpolation="nearest")
    pred_img = net_pp(
        masked_frames=masked_frames,
        completed_flows=(pred_flow_f, pred_flow_b),
        masks_in=masks_dilated,
        masks_updated=masks_dilated,
        num_local_frames=2)

    pred_img_np_ = pred_img[0].cpu().detach().numpy()

    # np.save(os.path.join(root_path, "pred_img.npy"), np.ascontiguousarray(pred_img_np_))

    pred_img_file_path = os.path.join(root_path, "pred_img.npy")
    pred_img_np = np.load(pred_img_file_path)

    if not np.array_equal(pred_img_np, pred_img_np_):
        print("*")
    np.testing.assert_array_equal(pred_img_np, pred_img_np_)

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
