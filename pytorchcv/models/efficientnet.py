"""
    EfficientNet for ImageNet-1K, implemented in PyTorch.
    Original papers:
    - 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946,
    - 'Adversarial Examples Improve Image Recognition,' https://arxiv.org/abs/1911.09665.
"""

__all__ = ['EfficientNet', 'calc_tf_padding', 'EffiInvResUnit', 'EffiInitBlock', 'efficientnet_b0', 'efficientnet_b1',
           'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6',
           'efficientnet_b7', 'efficientnet_b8', 'efficientnet_b0b', 'efficientnet_b1b', 'efficientnet_b2b',
           'efficientnet_b3b', 'efficientnet_b4b', 'efficientnet_b5b', 'efficientnet_b6b', 'efficientnet_b7b',
           'efficientnet_b0c', 'efficientnet_b1c', 'efficientnet_b2c', 'efficientnet_b3c', 'efficientnet_b4c',
           'efficientnet_b5c', 'efficientnet_b6c', 'efficientnet_b7c', 'efficientnet_b8c']

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from .common.common import (round_channels, lambda_swish, lambda_batchnorm2d, conv1x1_block, conv3x3_block, dwconv3x3_block,
                     dwconv5x5_block, SEBlock)


def calc_tf_padding(x: torch.Tensor,
                    kernel_size: int,
                    stride: int = 1,
                    dilation: int = 1) -> tuple[int, int, int, int]:
    """
    Calculate TF-same like padding size.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    kernel_size : int
        Convolution window size.
    stride : int, default 1
        Strides of the convolution.
    dilation : int, default 1
        Dilation value for convolution layer.

    Returns
    -------
    tuple(int, int, int, int)
        The size of the padding.
    """
    height, width = x.size()[2:]
    oh = math.ceil(float(height) / stride)
    ow = math.ceil(float(width) / stride)
    pad_h = max((oh - 1) * stride + (kernel_size - 1) * dilation + 1 - height, 0)
    pad_w = max((ow - 1) * stride + (kernel_size - 1) * dilation + 1 - width, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


class EffiDwsConvUnit(nn.Module):
    """
    EfficientNet specific depthwise separable convolution block/unit with BatchNorms and activations at each convolution
    layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the second convolution layer.
    normalization : function
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    tf_mode : bool
        Whether to use TF-like mode.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 normalization: Callable[..., nn.Module],
                 activation: Callable[..., nn.Module],
                 tf_mode: bool):
        super(EffiDwsConvUnit, self).__init__()
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)

        self.dw_conv = dwconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=(0 if tf_mode else 1),
            normalization=normalization,
            activation=activation)
        self.se = SEBlock(
            channels=in_channels,
            reduction=4,
            mid_activation=activation)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            normalization=normalization,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3))
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.pw_conv(x)
        if self.residual:
            x = x + identity
        return x


class EffiInvResUnit(nn.Module):
    """
    EfficientNet inverted residual unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    stride : int or tuple(int, int)
        Strides of the second convolution layer.
    exp_factor : int
        Factor for expansion of channels.
    se_factor : int
        SE reduction factor for each unit.
    normalization : function
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    tf_mode : bool
        Whether to use TF-like mode.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 exp_factor: int,
                 se_factor: int,
                 normalization: Callable[..., nn.Module],
                 activation: Callable[..., nn.Module],
                 tf_mode: bool):
        super(EffiInvResUnit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = in_channels * exp_factor
        dwconv_block_fn = dwconv3x3_block if kernel_size == 3 else (dwconv5x5_block if kernel_size == 5 else None)

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            normalization=normalization,
            activation=activation)
        self.conv2 = dwconv_block_fn(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=(0 if tf_mode else (kernel_size // 2)),
            normalization=normalization,
            activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=(exp_factor * se_factor),
                mid_activation=activation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            normalization=normalization,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=self.kernel_size, stride=self.stride))
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class EffiInitBlock(nn.Module):
    """
    EfficientNet specific initial block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    normalization : function
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    tf_mode : bool
        Whether to use TF-like mode.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 normalization: Callable[..., nn.Module],
                 activation: Callable[..., nn.Module],
                 tf_mode: bool):
        super(EffiInitBlock, self).__init__()
        self.tf_mode = tf_mode

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            padding=(0 if tf_mode else 1),
            normalization=normalization,
            activation=activation)

    def forward(self, x):
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3, stride=2))
        x = self.conv(x)
        return x


class EfficientNet(nn.Module):
    """
    EfficientNet model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernel_sizes : list(list(int))
        Number of kernel sizes for each unit.
    strides_per_stage : list(int)
        Stride value for the first unit of each stage.
    expansion_factors : list(list(int))
        Number of expansion factors for each unit.
    dropout_rate : float, default 0.2
        Fraction of the input units to drop. Must be a number between 0 and 1.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels: list[list[int]],
                 init_block_channels: int,
                 final_block_channels: int,
                 kernel_sizes: list[list[int]],
                 strides_per_stage: list[int],
                 expansion_factors: list[list[int]],
                 dropout_rate: float = 0.2,
                 tf_mode: bool = False,
                 bn_eps: float = 1e-5,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (224, 224),
                 num_classes: int = 1000):
        super(EfficientNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        normalization = lambda_batchnorm2d(eps=bn_eps)
        activation = lambda_swish()

        self.features = nn.Sequential()
        self.features.add_module("init_block", EffiInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            normalization=normalization,
            activation=activation,
            tf_mode=tf_mode))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            kernel_sizes_per_stage = kernel_sizes[i]
            expansion_factors_per_stage = expansion_factors[i]
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expansion_factor = expansion_factors_per_stage[j]
                stride = strides_per_stage[i] if (j == 0) else 1
                if i == 0:
                    stage.add_module("unit{}".format(j + 1), EffiDwsConvUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        normalization=normalization,
                        activation=activation,
                        tf_mode=tf_mode))
                else:
                    stage.add_module("unit{}".format(j + 1), EffiInvResUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        exp_factor=expansion_factor,
                        se_factor=4,
                        normalization=normalization,
                        activation=activation,
                        tf_mode=tf_mode))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            normalization=normalization,
            activation=activation))
        in_channels = final_block_channels
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.output = nn.Sequential()
        if dropout_rate > 0.0:
            self.output.add_module("dropout", nn.Dropout(p=dropout_rate))
        self.output.add_module("fc", nn.Linear(
            in_features=in_channels,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_efficientnet(version: str,
                     in_size: tuple[int, int],
                     tf_mode: bool = False,
                     bn_eps: float = 1e-5,
                     model_name: str | None = None,
                     pretrained: bool = False,
                     root: str = os.path.join("~", ".torch", "models"),
                     **kwargs) -> nn.Module:
    """
    Create EfficientNet model with specific parameters.

    Parameters
    ----------
    version : str
        Version of EfficientNet ('b0'...'b8').
    in_size : tuple(int, int)
        Spatial size of the expected input image.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
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
    if version == "b0":
        assert (in_size == (224, 224))
        depth_factor = 1.0
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b1":
        assert (in_size == (240, 240))
        depth_factor = 1.1
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b2":
        assert (in_size == (260, 260))
        depth_factor = 1.2
        width_factor = 1.1
        dropout_rate = 0.3
    elif version == "b3":
        assert (in_size == (300, 300))
        depth_factor = 1.4
        width_factor = 1.2
        dropout_rate = 0.3
    elif version == "b4":
        assert (in_size == (380, 380))
        depth_factor = 1.8
        width_factor = 1.4
        dropout_rate = 0.4
    elif version == "b5":
        assert (in_size == (456, 456))
        depth_factor = 2.2
        width_factor = 1.6
        dropout_rate = 0.4
    elif version == "b6":
        assert (in_size == (528, 528))
        depth_factor = 2.6
        width_factor = 1.8
        dropout_rate = 0.5
    elif version == "b7":
        assert (in_size == (600, 600))
        depth_factor = 3.1
        width_factor = 2.0
        dropout_rate = 0.5
    elif version == "b8":
        assert (in_size == (672, 672))
        depth_factor = 3.6
        width_factor = 2.2
        dropout_rate = 0.5
    else:
        raise ValueError("Unsupported EfficientNet version {}".format(version))

    init_block_channels = 32
    layers = [1, 2, 2, 3, 3, 4, 1]
    downsample = [1, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
    expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
    strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
    final_block_channels = 1280

    layers = [int(math.ceil(li * depth_factor)) for li in layers]
    channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    kernel_sizes = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                          zip(kernel_sizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])
    strides_per_stage = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(strides_per_stage, layers, downsample), [])
    strides_per_stage = [si[0] for si in strides_per_stage]

    init_block_channels = round_channels(init_block_channels * width_factor)

    if width_factor > 1.0:
        assert (int(final_block_channels * width_factor) == round_channels(final_block_channels * width_factor))
        final_block_channels = round_channels(final_block_channels * width_factor)

    net = EfficientNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        strides_per_stage=strides_per_stage,
        expansion_factors=expansion_factors,
        dropout_rate=dropout_rate,
        tf_mode=tf_mode,
        bn_eps=bn_eps,
        in_size=in_size,
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


def efficientnet_b0(in_size: tuple[int, int] = (224, 224),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B0 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b0",
        in_size=in_size,
        model_name="efficientnet_b0",
        **kwargs)


def efficientnet_b1(in_size: tuple[int, int] = (240, 240),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B1 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (240, 240)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b1",
        in_size=in_size,
        model_name="efficientnet_b1",
        **kwargs)


def efficientnet_b2(in_size: tuple[int, int] = (260, 260),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B2 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (260, 260)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b2",
        in_size=in_size,
        model_name="efficientnet_b2",
        **kwargs)


def efficientnet_b3(in_size: tuple[int, int] = (300, 300),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B3 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (300, 300)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b3",
        in_size=in_size,
        model_name="efficientnet_b3",
        **kwargs)


def efficientnet_b4(in_size: tuple[int, int] = (380, 380),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B4 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (380, 380)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b4",
        in_size=in_size,
        model_name="efficientnet_b4",
        **kwargs)


def efficientnet_b5(in_size: tuple[int, int] = (456, 456),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B5 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (456, 456)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b5",
        in_size=in_size,
        model_name="efficientnet_b5",
        **kwargs)


def efficientnet_b6(in_size: tuple[int, int] = (528, 528),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B6 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (528, 528)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b6",
        in_size=in_size,
        model_name="efficientnet_b6",
        **kwargs)


def efficientnet_b7(in_size: tuple[int, int] = (600, 600),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B7 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (600, 600)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b7",
        in_size=in_size,
        model_name="efficientnet_b7",
        **kwargs)


def efficientnet_b8(in_size: tuple[int, int] = (672, 672),
                    **kwargs) -> nn.Module:
    """
    EfficientNet-B8 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (672, 672)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b8",
        in_size=in_size,
        model_name="efficientnet_b8",
        **kwargs)


def efficientnet_b0b(in_size: tuple[int, int] = (224, 224),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B0-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b0",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b0b",
        **kwargs)


def efficientnet_b1b(in_size: tuple[int, int] = (240, 240),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B1-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (240, 240)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b1",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b1b",
        **kwargs)


def efficientnet_b2b(in_size: tuple[int, int] = (260, 260),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B2-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (260, 260)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b2",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b2b",
        **kwargs)


def efficientnet_b3b(in_size: tuple[int, int] = (300, 300),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B3-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (300, 300)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b3",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b3b",
        **kwargs)


def efficientnet_b4b(in_size: tuple[int, int] = (380, 380),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B4-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (380, 380)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b4",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b4b",
        **kwargs)


def efficientnet_b5b(in_size: tuple[int, int] = (456, 456),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B5-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (456, 456)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b5",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b5b",
        **kwargs)


def efficientnet_b6b(in_size: tuple[int, int] = (528, 528),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B6-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (528, 528)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b6",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b6b",
        **kwargs)


def efficientnet_b7b(in_size: tuple[int, int] = (600, 600),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B7-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (600, 600)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b7",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b7b",
        **kwargs)


def efficientnet_b0c(in_size: tuple[int, int] = (224, 224),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B0-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b0",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b0c",
        **kwargs)


def efficientnet_b1c(in_size: tuple[int, int] = (240, 240),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B1-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (240, 240)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b1",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b1c",
        **kwargs)


def efficientnet_b2c(in_size: tuple[int, int] = (260, 260),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B2-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (260, 260)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b2",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b2c",
        **kwargs)


def efficientnet_b3c(in_size: tuple[int, int] = (300, 300),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B3-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (300, 300)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b3",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b3c",
        **kwargs)


def efficientnet_b4c(in_size: tuple[int, int] = (380, 380),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B4-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (380, 380)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b4",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b4c",
        **kwargs)


def efficientnet_b5c(in_size: tuple[int, int] = (456, 456),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B5-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (456, 456)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b5",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b5c",
        **kwargs)


def efficientnet_b6c(in_size: tuple[int, int] = (528, 528),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B6-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (528, 528)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b6",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b6c",
        **kwargs)


def efficientnet_b7c(in_size: tuple[int, int] = (600, 600),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B7-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (600, 600)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b7",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b7c",
        **kwargs)


def efficientnet_b8c(in_size: tuple[int, int] = (672, 672),
                     **kwargs) -> nn.Module:
    """
    EfficientNet-B8-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.

    Parameters
    ----------
    in_size : tuple(int, int), default (672, 672)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_efficientnet(
        version="b8",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b8c",
        **kwargs)


def _test():
    from .model_store import calc_net_weight_count

    pretrained = False

    models = [
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7,
        efficientnet_b8,
        efficientnet_b0b,
        efficientnet_b1b,
        efficientnet_b2b,
        efficientnet_b3b,
        efficientnet_b4b,
        efficientnet_b5b,
        efficientnet_b6b,
        efficientnet_b7b,
        efficientnet_b0c,
        efficientnet_b1c,
        efficientnet_b2c,
        efficientnet_b3c,
        efficientnet_b4c,
        efficientnet_b5c,
        efficientnet_b6c,
        efficientnet_b7c,
        efficientnet_b8c,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != efficientnet_b0 or weight_count == 5288548)
        assert (model != efficientnet_b1 or weight_count == 7794184)
        assert (model != efficientnet_b2 or weight_count == 9109994)
        assert (model != efficientnet_b3 or weight_count == 12233232)
        assert (model != efficientnet_b4 or weight_count == 19341616)
        assert (model != efficientnet_b5 or weight_count == 30389784)
        assert (model != efficientnet_b6 or weight_count == 43040704)
        assert (model != efficientnet_b7 or weight_count == 66347960)
        assert (model != efficientnet_b8 or weight_count == 87413142)
        assert (model != efficientnet_b0b or weight_count == 5288548)
        assert (model != efficientnet_b1b or weight_count == 7794184)
        assert (model != efficientnet_b2b or weight_count == 9109994)
        assert (model != efficientnet_b3b or weight_count == 12233232)
        assert (model != efficientnet_b4b or weight_count == 19341616)
        assert (model != efficientnet_b5b or weight_count == 30389784)
        assert (model != efficientnet_b6b or weight_count == 43040704)
        assert (model != efficientnet_b7b or weight_count == 66347960)
        assert (model != efficientnet_b0c or weight_count == 5288548)
        assert (model != efficientnet_b1c or weight_count == 7794184)
        assert (model != efficientnet_b2c or weight_count == 9109994)
        assert (model != efficientnet_b3c or weight_count == 12233232)
        assert (model != efficientnet_b4c or weight_count == 19341616)
        assert (model != efficientnet_b5c or weight_count == 30389784)
        assert (model != efficientnet_b6c or weight_count == 43040704)
        assert (model != efficientnet_b7c or weight_count == 66347960)

        x = torch.randn(1, 3, net.in_size[0], net.in_size[1])
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
