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
from common import (lambda_relu, create_activation_layer, lambda_leakyrelu, lambda_tanh, conv1x1, conv3x3,
                    conv3x3_block, InterpolationBlock)
from resnet import ResUnit, ResBlock
from propainter_rfc import SecondOrderDeformableAlignment
from propainter_ip import propainter_ip, BidirectionalPropagation


class Encoder(nn.Module):
    """
    Encoder unit.

    Parameters
    ----------
    activation : function
        Lambda-function generator for activation layer in convolution blocks.
    """
    def __init__(self,
                 activation: Callable[..., nn.Module]):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            conv3x3_block(
                in_channels=5,
                out_channels=64,
                stride=2,
                bias=True,
                normalization=None,
                activation=activation),
            conv3x3_block(
                in_channels=64,
                out_channels=64,
                bias=True,
                normalization=None,
                activation=activation),
            conv3x3_block(
                in_channels=64,
                out_channels=128,
                stride=2,
                bias=True,
                normalization=None,
                activation=activation),
            conv3x3_block(
                in_channels=128,
                out_channels=256,
                bias=True,
                normalization=None,
                activation=activation),
            conv3x3_block(
                in_channels=256,
                out_channels=384,
                groups=1,
                bias=True,
                normalization=None,
                activation=activation),
            conv3x3_block(
                in_channels=640,
                out_channels=512,
                groups=2,
                bias=True,
                normalization=None,
                activation=activation),
            conv3x3_block(
                in_channels=768,
                out_channels=384,
                groups=4,
                bias=True,
                normalization=None,
                activation=activation),
            conv3x3_block(
                in_channels=640,
                out_channels=256,
                groups=8,
                bias=True,
                normalization=None,
                activation=activation),
            conv3x3_block(
                in_channels=512,
                out_channels=128,
                groups=1,
                bias=True,
                normalization=None,
                activation=activation),
        ])

    def forward(self, x):
        batch, _, x_height, x_width = x.size()
        out = x
        for i, layer in enumerate(self.layers):
            if i == 4:
                x0 = out
                _, _, height, width = x0.size()
                assert (height == x_height // 4)
                assert (width == x_width // 4)
            if i > 4:
                g = self.group[i - 4]
                assert (g == layer.conv.groups)
                y = x0.view(batch, g, -1, height, width)
                o = out.view(batch, g, -1, height, width)
                out = torch.cat([y, o], 2).view(batch, -1, height, width)
            out = layer(out)
        return out


class PPDecoderUnit(nn.Module):
    """
    Decoder unit (specific for ProPainter).

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
        super(PPDecoderUnit, self).__init__()
        self.up = InterpolationBlock(scale_factor=2)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            bias=True,
            normalization=None,
            activation=activation)
        self.conv2 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            normalization=None,
            activation=final_activation)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """
    Decoder unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    activation : function
        Lambda-function generator for activation layer in convolution blocks.
    final_activation : function or None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 activation: Callable[..., nn.Module],
                 final_activation: Callable[..., nn.Module | None] | None):
        super(Decoder, self).__init__()
        self.unit1 = PPDecoderUnit(
            in_channels=in_channels,
            out_channels=64,
            activation=activation,
            final_activation=activation)
        self.unit2 = PPDecoderUnit(
            in_channels=64,
            out_channels=3,
            activation=activation,
            final_activation=final_activation)
        # self.body = nn.Sequential(
        #     InterpolationBlock(scale_factor=2),
        #     conv3x3_block(
        #         in_channels=in_channels,
        #         out_channels=128,
        #         bias=True,
        #         normalization=None,
        #         activation=activation),
        #     conv3x3_block(
        #         in_channels=128,
        #         out_channels=64,
        #         bias=True,
        #         normalization=None,
        #         activation=activation),
        #     InterpolationBlock(scale_factor=2),
        #     conv3x3_block(
        #         in_channels=64,
        #         out_channels=64,
        #         bias=True,
        #         normalization=None,
        #         activation=activation),
        #     conv3x3_block(
        #         in_channels=64,
        #         out_channels=3,
        #         bias=True,
        #         normalization=None,
        #         activation=final_activation))

    def forward(self, x):
        x = self.unit1(x)
        x = self.unit2(x)
        return x


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.up = InterpolationBlock(scale_factor=2)
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


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
        channels = 128
        hidden = 512
        activation = lambda_leakyrelu(negative_slope=0.2)

        # encoder
        self.encoder = Encoder(activation=activation)

        # decoder
        self.decoder = Decoder(
            in_channels=channels,
            activation=activation,
            final_activation=lambda_tanh())

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
        self.ss = SoftSplit(channels, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channels, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        # feature propagation module
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
        batch, time, _, orig_height, orig_width = masked_frames.size()

        # Extracting features:
        enc_feat = self.encoder(torch.cat([
            masked_frames.view(batch * time, 3, orig_height, orig_width),
            masks_in.view(batch * time, 1, orig_height, orig_width),
            masks_updated.view(batch * time, 1, orig_height, orig_width)], dim=1))
        _, channels, height, width = enc_feat.size()
        local_feat = enc_feat.view(batch, time, channels, height, width)[:, :l_t, ...]
        ref_feat = enc_feat.view(batch, time, channels, height, width)[:, l_t:, ...]
        fold_feat_size = (height, width)

        ds_flows_f = F.interpolate(
            input=completed_flows[0].view(-1, 2, orig_height, orig_width),
            scale_factor=(1 / 4),
            mode="bilinear",
            align_corners=False).view(batch, l_t - 1, 2, height, width) / 4.0
        ds_flows_b = F.interpolate(
            completed_flows[1].view(-1, 2, orig_height, orig_width),
            scale_factor=(1 / 4),
            mode="bilinear",
            align_corners=False).view(batch, l_t - 1, 2, height, width) / 4.0
        ds_mask_in = F.interpolate(
            masks_in.reshape(-1, 1, orig_height, orig_width),
            scale_factor=(1 / 4),
            mode="nearest").view(batch, time, 1, height, width)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local = F.interpolate(
            masks_updated[:, :l_t].reshape(-1, 1, orig_height, orig_width),
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
            output = output.view(batch, time, 3, orig_height, orig_width)
        else:
            output = self.decoder(enc_feat[:, :l_t].view(-1, channels, height, width))
            output = output.view(batch, l_t, 3, orig_height, orig_width)

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

        list3 = list(filter(re.compile("feat_prop_module.fuse.").search, src_param_keys))
        list3_u = [key.replace(".0.", ".conv1.conv.") for key in list3]
        list3_u = [key.replace(".2.", ".conv2.conv.") for key in list3_u]
        for src_i, dst_i in zip(list3, list3_u):
            upd_dict[src_i] = dst_i

        list8 = list(filter(re.compile("feat_prop_module.backbone.").search, src_param_keys))
        list8_u = [key.replace(".0.", ".conv1.conv.") for key in list8]
        list8_u = [key.replace(".2.", ".conv2.conv.") for key in list8_u]
        for src_i, dst_i in zip(list8, list8_u):
            upd_dict[src_i] = dst_i

        list4 = list(filter(re.compile("encoder.layers.").search, src_param_keys))
        list4_u = [key.replace(".0.", ".0.conv.") for key in list4]
        list4_u = [key.replace(".2.", ".1.conv.") for key in list4_u]
        list4_u = [key.replace(".4.", ".2.conv.") for key in list4_u]
        list4_u = [key.replace(".6.", ".3.conv.") for key in list4_u]
        list4_u = [key.replace(".8.", ".4.conv.") for key in list4_u]
        list4_u = [key.replace(".10.", ".5.conv.") for key in list4_u]
        list4_u = [key.replace(".12.", ".6.conv.") for key in list4_u]
        list4_u = [key.replace(".14.", ".7.conv.") for key in list4_u]
        list4_u = [key.replace(".16.", ".8.conv.") for key in list4_u]
        for src_i, dst_i in zip(list4, list4_u):
            upd_dict[src_i] = dst_i

        list4 = list(filter(re.compile("decoder.").search, src_param_keys))
        list4_u = [key.replace(".0.", ".unit1.conv1.") for key in list4]
        list4_u = [key.replace(".2.", ".unit1.conv2.conv.") for key in list4_u]
        list4_u = [key.replace(".4.", ".unit2.conv1.") for key in list4_u]
        list4_u = [key.replace(".6.", ".unit2.conv2.conv.") for key in list4_u]
        for src_i, dst_i in zip(list4, list4_u):
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

    # prop_imgs, updated_local_masks = net_pp.img_propagation(
    #     masked_frames=masked_frames,
    #     completed_flows=(pred_flow_f, pred_flow_b),
    #     masks=masks_dilated,
    #     interpolation="nearest")

    ppip_net = propainter_ip()
    ppip_net.eval()
    ppip_net = ppip_net.cuda()
    comp_flows = torch.cat((pred_flow_f[0], pred_flow_b[0]), dim=1)
    prop_imgs, updated_local_masks = ppip_net(
        frames=frames[0],
        masks=masks_dilated[0],
        comp_flows=comp_flows,
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
