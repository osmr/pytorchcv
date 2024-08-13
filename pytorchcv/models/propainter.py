"""
    ProPainter for video inpainting, implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['ProPainter', 'propainter']

import os
import math
import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from typing import Callable
from .common.activ import lambda_leakyrelu, lambda_tanh
from .common.conv import conv3x3, conv3x3_block
from .common.tutti import InterpolationBlock
from .propainter_ip import BidirectionalPropagation


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
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    activation : function
        Lambda-function generator for activation layer in convolution blocks.
    final_activation : function or None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 activation: Callable[..., nn.Module],
                 final_activation: Callable[..., nn.Module | None] | None):
        super(Decoder, self).__init__()
        self.unit1 = PPDecoderUnit(
            in_channels=in_channels,
            out_channels=mid_channels,
            activation=activation,
            final_activation=activation)
        self.unit2 = PPDecoderUnit(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=activation,
            final_activation=final_activation)

    def forward(self, x):
        x = self.unit1(x)
        x = self.unit2(x)
        return x


class SoftSplit(nn.Module):
    """
    Soft split.
    """
    def __init__(self,
                 channels: int,
                 hidden_dim: int,
                 kernel_size: tuple[int, int],
                 stride: tuple[int, int],
                 padding: tuple[int, int]):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.t2t = nn.Unfold(
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

        emb_in_channels = reduce((lambda x, y: x * y), kernel_size) * channels
        self.embedding = nn.Linear(
            in_features=emb_in_channels,
            out_features=hidden_dim)

    def forward(self,
                x: torch.Tensor,
                batch: int,
                output_size: tuple[int, int]):
        f_height = int((output_size[0] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_width = int((output_size[1] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = self.t2t(x)
        x = x.permute(0, 2, 1)
        # x shape [b*t, num_vec, ks*ks*c]
        x = self.embedding(x)

        # x shape after embedding [b, t*num_vec, hidden]
        x = x.view(batch, -1, f_height, f_width, x.size(2))
        return x


class SoftComp(nn.Module):
    """
    Soft composition.
    """
    def __init__(self,
                 channels: int,
                 hidden_dim: int,
                 kernel_size: tuple[int, int],
                 stride: tuple[int, int],
                 padding: tuple[int, int]):
        super(SoftComp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        emb_out_channels = reduce((lambda x, y: x * y), kernel_size) * channels
        self.embedding = nn.Linear(
            in_features=hidden_dim,
            out_features=emb_out_channels)

        self.bias_conv = conv3x3(
            in_channels=channels,
            out_channels=channels,
            bias=True)

    def forward(self,
                x: torch.Tensor,
                time: int,
                output_size: tuple[int, int]):
        batch, _, _, _, x_channels = x.shape
        x = x.view(batch, -1, x_channels)

        y = self.embedding(x)

        _, _, y_channels = y.size()
        y = y.view(batch * time, -1, y_channels).permute(0, 2, 1)

        y = F.fold(
            input=y,
            output_size=output_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride)
        y = self.bias_conv(y)
        return y


def window_partition(x: torch.Tensor,
                     window_size: tuple[int, int],
                     num_heads: int):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, num_heads, T, window_size, window_size, C//num_heads)
    """
    batch, time, height, width, channels = x.shape
    win_height, win_width = window_size
    x = x.view(
        batch,
        time,
        height // win_height,
        win_height,
        width // win_width,
        win_width,
        num_heads,
        channels // num_heads)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows


class SparseWindowAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: tuple[int, int],
                 pool_size: tuple[int, int] = (4, 4),
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 pooling_token: bool = True):
        super(SparseWindowAttention, self).__init__()
        assert (dim % num_heads == 0)

        # Key, query, value projections for all heads:
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)

        # Regularization:
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Output projection:
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=ks,
                stride=stride,
                padding=(0, 0),
                groups=dim)
            self.pool_layer.weight.data.fill_(1.0 / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
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

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                time_idx=None):
        b, t, h, w, c = x.shape  # 20 36
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.num_heads
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0]  # 20
        new_w = n_ww * self.window_size[1]  # 36
        pad_r = new_w - w
        pad_b = new_h - h
        # reverse order
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)
            mask = F.pad(mask, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        win_q = window_partition(q.contiguous(), self.window_size, self.num_heads).view(
            b, n_wh * n_ww, self.num_heads, t, w_h * w_w, c_head)
        win_k = window_partition(k.contiguous(), self.window_size, self.num_heads).view(
            b, n_wh * n_ww, self.num_heads, t, w_h * w_w, c_head)
        win_v = window_partition(v.contiguous(), self.window_size, self.num_heads).view(
            b, n_wh * n_ww, self.num_heads, t, w_h * w_w, c_head)
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
                lambda a: window_partition(a, self.window_size, self.num_heads).view(
                    b, n_wh * n_ww, self.num_heads, t, w_h * w_w, c_head),
                (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda a: window_partition(a, self.window_size, self.num_heads).view(
                    b, n_wh * n_ww, self.num_heads, t, w_h * w_w, c_head),
                (v_tl, v_tr, v_bl, v_br))
            rool_k = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), dim=4).contiguous()
            rool_v = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), dim=4).contiguous()  # [b, n_wh*n_ww, num_heads, t, w_h*w_w, c_head]  # noqa
            # mask out tokens in current window
            rool_k = rool_k[:, :, :, :, self.valid_ind_rolled]
            rool_v = rool_v[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_k.shape[4]
            rool_k = rool_k.view(b, n_wh * n_ww, self.num_heads, t, roll_N, c // self.num_heads)
            rool_v = rool_v.view(b, n_wh * n_ww, self.num_heads, t, roll_N, c // self.num_heads)
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
            pool_k = pool_k.view(b, n_wh * n_ww, t, p_h, p_w, self.num_heads, c_head).permute(0, 1, 5, 2, 3, 4, 6)
            pool_k = pool_k.contiguous().view(b, n_wh * n_ww, self.num_heads, t, p_h * p_w, c_head)
            win_k = torch.cat((win_k, pool_k), dim=4)
            # pool_v
            pool_v = self.value(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)  # [b, n_wh*n_ww, t, p_h, p_w, c]  # noqa
            pool_v = pool_v.view(b, n_wh * n_ww, t, p_h, p_w, self.num_heads, c_head).permute(0, 1, 5, 2, 3, 4, 6)
            pool_v = pool_v.contiguous().view(b, n_wh * n_ww, self.num_heads, t, p_h * p_w, c_head)
            win_v = torch.cat((win_v, pool_v), dim=4)

        # [b, n_wh*n_ww, num_heads, t, w_h*w_w, c_head]
        out = torch.zeros_like(win_q)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(b * l_t, new_h, new_w))
        mask = mask.view(b, l_t, n_wh * n_ww)
        mask = torch.sum(mask, dim=1)  # [b, n_wh*n_ww]
        for i in range(win_q.shape[0]):
            # For masked windows:
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, num_heads, t, w_h*w_w, c_head]
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(mask_n, self.num_heads, t * w_h * w_w, c_head)
                win_k_t = win_k[i, mask_ind_i]
                win_v_t = win_v[i, mask_ind_i]
                # mask out key and value
                if time_idx is not None:
                    # key [n_wh*n_ww, num_heads, t, w_h*w_w, c_head]
                    win_k_t = win_k_t[:, :, time_idx.view(-1)].view(mask_n, self.num_heads, -1, c_head)
                    # value
                    win_v_t = win_v_t[:, :, time_idx.view(-1)].view(mask_n, self.num_heads, -1, c_head)
                else:
                    win_k_t = win_k_t.view(n_wh * n_ww, self.num_heads, t * w_h * w_w, c_head)
                    win_v_t = win_v_t.view(n_wh * n_ww, self.num_heads, t * w_h * w_w, c_head)

                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                att_t = F.softmax(att_t, dim=-1)
                att_t = self.attn_drop(att_t)
                y_t = att_t @ win_v_t

                out[i, mask_ind_i] = y_t.view(-1, self.num_heads, t, w_h * w_w, c_head)

            # For unmasked windows:
            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, num_heads, t, w_h*w_w, c_head]
            win_q_s = win_q[i, unmask_ind_i]
            win_k_s = win_k[i, unmask_ind_i, :, :, :w_h * w_w]
            win_v_s = win_v[i, unmask_ind_i, :, :, :w_h * w_w]

            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            y_s = att_s @ win_v_s
            out[i, unmask_ind_i] = y_s

        # Re-assemble all head outputs side by side:
        out = out.view(b, n_wh, n_ww, self.num_heads, t, w_h, w_w, c_head)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)

        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]

        # Output projection:
        out = self.proj_drop(self.proj(out))
        return out


class FusionFeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 kernel_size: tuple[int, int],
                 stride: tuple[int, int],
                 padding: tuple[int, int]):
        super(FusionFeedForward, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=dim,
                out_features=hidden_dim))
        self.fc2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=dim))

        self.kernel_shape = reduce((lambda x, y: x * y), self.kernel_size)  # 49

    def forward(self,
                x: torch.Tensor,
                output_size: tuple[int, int]):
        n_vecs = 1
        for i, d in enumerate(self.kernel_size):
            n_vecs *= int((output_size[i] + 2 * self.padding[i] - (d - 1) - 1) / self.stride[i] + 1)

        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(
            input=normalizer,
            output_size=output_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride)
        x = F.fold(
            input=x.view(-1, n_vecs, c).permute(0, 2, 1),
            output_size=output_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride)
        x = F.unfold(
            input=(x / normalizer),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride)
        x = x.permute(0, 2, 1).contiguous().view(b, n, c)

        x = self.fc2(x)
        return x


class TemporalSparseTransformer(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: tuple[int, int],
                 pool_size: tuple[int, int],
                 kernel_size: tuple[int, int],
                 stride: tuple[int, int],
                 padding: tuple[int, int]):
        super(TemporalSparseTransformer, self).__init__()
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(normalized_shape=dim)
        self.attention = SparseWindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pool_size=pool_size)

        self.norm2 = nn.LayerNorm(normalized_shape=dim)
        self.mlp = FusionFeedForward(
            dim=dim,
            hidden_dim=1960,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self,
                x: torch.Tensor,
                fold_x_size: tuple[int, int],
                mask: torch.Tensor,
                time_idx=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        batch, time, height, width, channels = x.shape  # 20 36

        y = self.norm1(x)
        y = self.attention(x=y, mask=mask, time_idx=time_idx)
        x = x + y

        # FFN
        y = self.norm2(x)
        y = y.view(batch, time * height * width, channels)
        y = self.mlp(y, fold_x_size)
        y = y.view(batch, time, height, width, channels)
        x = x + y

        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: tuple[int, int],
                 pool_size: tuple[int, int],
                 kernel_size: tuple[int, int],
                 stride: tuple[int, int],
                 padding: tuple[int, int],
                 depth: int):
        super(TemporalSparseTransformerBlock, self).__init__()
        self.depth = depth

        blocks = []
        for i in range(depth):
            blocks.append(TemporalSparseTransformer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                pool_size=pool_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))
        self.transformer = nn.Sequential(*blocks)

    def forward(self,
                x: torch.Tensor,
                fold_x_size: tuple[int, int],
                l_mask: torch.Tensor,
                time_dilation: int = 2):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            l_mask: local mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        assert (self.depth % time_dilation == 0)

        time = x.size(1)
        time_idx = [torch.arange(i, time, time_dilation) for i in range(time_dilation)] * (self.depth // time_dilation)

        for i in range(0, self.depth):
            x = self.transformer[i](x, fold_x_size, l_mask, time_idx[i])

        return x


class ProPainter(nn.Module):
    def __init__(self,
                 channels: int = 128,
                 hidden_dim: int = 512,
                 num_heads: int = 4,
                 depth: int = 8,
                 t2t_kernel_size: tuple[int, int] = (7, 7),
                 t2t_padding: tuple[int, int] = (3, 3),
                 t2t_stride: tuple[int, int] = (3, 3),
                 window_size: tuple[int, int] = (5, 9),
                 pool_size: tuple[int, int] = (4, 4)):
        super(ProPainter, self).__init__()
        activation = lambda_leakyrelu(negative_slope=0.2)

        self.encoder = Encoder(activation=activation)

        self.decoder = Decoder(
            in_channels=channels,
            mid_channels=64,
            out_channels=3,
            activation=activation,
            final_activation=lambda_tanh())

        self.ss = SoftSplit(
            channels=channels,
            hidden_dim=hidden_dim,
            kernel_size=t2t_kernel_size,
            stride=t2t_stride,
            padding=t2t_padding)
        self.sc = SoftComp(
            channels=channels,
            hidden_dim=hidden_dim,
            kernel_size=t2t_kernel_size,
            stride=t2t_stride,
            padding=t2t_padding)
        self.max_pool = nn.MaxPool2d(
            kernel_size=t2t_kernel_size,
            stride=t2t_stride,
            padding=t2t_padding)

        # Feature propagation module
        self.feat_prop_module = BidirectionalPropagation(
            channels=channels,
            learnable=True)

        self.transformers = TemporalSparseTransformerBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            pool_size=pool_size,
            kernel_size=t2t_kernel_size,
            stride=t2t_stride,
            padding=t2t_padding,
            depth=depth)

        self._init_weights()

    def _init_weights(self,
                      init_type="normal",
                      gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("InstanceNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "none":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError("Initialization method [%s] is not implemented" % init_type)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)

    def forward(self,
                masked_frames: torch.Tensor,
                masks_updated: torch.Tensor,
                masks_in: torch.Tensor,
                completed_flows: torch.Tensor,
                num_local_frames: int,
                interpolation: str = "bilinear",
                time_dilation: int = 2):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """
        comp_flows_forward, comp_flows_backward = torch.split(completed_flows, [2, 2], dim=2)

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
            input=comp_flows_forward.view(-1, 2, orig_height, orig_width),
            scale_factor=(1 / 4),
            mode="bilinear",
            align_corners=False).view(batch, l_t - 1, 2, height, width) / 4.0
        ds_flows_b = F.interpolate(
            input=comp_flows_backward.view(-1, 2, orig_height, orig_width),
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
        # from einops import rearrange
        # mask_pool_l = rearrange(mask_pool_l, "b t c h w -> b t h w c").contiguous()
        mask_pool_l = mask_pool_l.permute(0, 1, 3, 4, 2).contiguous()
        trans_feat = self.transformers(
            x=trans_feat,
            fold_x_size=fold_feat_size,
            l_mask=mask_pool_l,
            time_dilation=time_dilation)
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


def get_propainter(model_name: str | None = None,
                   pretrained: bool = False,
                   root: str = os.path.join("~", ".torch", "models"),
                   **kwargs) -> nn.Module:
    """
    Create ProPainter model with specific parameters.

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
    net = ProPainter(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .common.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def propainter(**kwargs) -> nn.Module:
    """
    ProPainter model from 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
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
    return get_propainter(
        model_name="propainter",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        propainter,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != propainter or weight_count == 39429667)

        batch = 4
        time = 5
        height = 240
        width = 432
        x1 = torch.randn(batch, time, 2, height, width)
        x2 = torch.randn(batch, time, 1, height, width)
        pred_frame = net(x1, x2)
        # y1.sum().backward()
        # y2.sum().backward()
        assert (tuple(pred_frame.size()) == (batch, time, 2, height, width))


if __name__ == "__main__":
    _test()
