"""
    PRNet for AFLW2000-3D, implemented in PyTorch.
    Original paper: 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network,'
    https://arxiv.org/abs/1803.07835.
"""

__all__ = ['PRNet', 'prnet']

import os
import torch.nn as nn
from typing import Callable
from .common.activ import lambda_relu
from .common.norm import lambda_batchnorm2d
from .common.conv import ConvBlock, DeconvBlock, conv1x1, conv1x1_block
from .common.tutti import NormActivation


def conv4x4_block(in_channels: int,
                  out_channels: int,
                  stride: int | tuple[int, int] = 1,
                  padding: int | tuple[int, int] | tuple[int, int, int, int] = (1, 2, 1, 2),
                  dilation: int | tuple[int, int] = 1,
                  groups: int = 1,
                  bias: bool = False,
                  normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                  activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu()):
    """
    4x4 version of the standard convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int) or tuple(int, int, int, int), default (1, 2, 1, 2)
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        normalization=normalization,
        activation=activation)


def deconv4x4_block(in_channels: int,
                    out_channels: int,
                    stride: int | tuple[int, int] = 1,
                    padding: int | tuple[int, int] | tuple[int, int, int, int] = 3,
                    ext_padding: tuple[int, int, int, int] = (2, 1, 2, 1),
                    out_padding: int | tuple[int, int] = 0,
                    dilation: int | tuple[int, int] = 1,
                    groups: int = 1,
                    bias: bool = False,
                    normalization: Callable[..., nn.Module | None] | nn.Module | None = lambda_batchnorm2d(),
                    activation: Callable[..., nn.Module | None] | nn.Module | str | None = lambda_relu()):
    """
    4x4 version of the standard deconvolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int), default 3
        Padding value for deconvolution layer.
    ext_padding : tuple(int, int, int, int), default (2, 1, 2, 1)
        Extra padding value for deconvolution layer.
    out_padding : int or tuple(int, int)
        Output padding value for deconvolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or nn.Module or None, default lambda_batchnorm2d()
        Lambda-function generator or module for normalization layer.
    activation : function or nn.Module or str or None, default lambda_relu()
        Lambda-function generator or module for activation layer.
    """
    return DeconvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=stride,
        padding=padding,
        ext_padding=ext_padding,
        out_padding=out_padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        normalization=normalization,
        activation=activation)


class PRResBottleneck(nn.Module):
    """
    PRNet specific bottleneck block for residual path in residual unit unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int)
        Padding value for the second convolution layer in bottleneck.
    normalization : function
        Lambda-function generator for normalization layer.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 normalization: Callable[..., nn.Module],
                 bottleneck_factor: int = 2):
        super(PRResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            normalization=normalization)
        self.conv2 = conv4x4_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=padding,
            normalization=normalization)
        self.conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PRResUnit(nn.Module):
    """
    PRNet specific ResNet unit with residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int)
        Padding value for the second convolution layer in bottleneck.
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 normalization: Callable[..., nn.Module]):
        super(PRResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        self.body = PRResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
            normalization=normalization)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            normalization=normalization)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.norm_activ(x)
        return x


class PROutputBlock(nn.Module):
    """
    PRNet specific output block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 normalization: Callable[..., nn.Module]):
        super(PROutputBlock, self).__init__()
        self.conv1 = deconv4x4_block(
            in_channels=in_channels,
            out_channels=out_channels,
            normalization=normalization)
        self.conv2 = deconv4x4_block(
            in_channels=out_channels,
            out_channels=out_channels,
            normalization=normalization)
        self.conv3 = deconv4x4_block(
            in_channels=out_channels,
            out_channels=out_channels,
            normalization=normalization,
            activation=nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PRNet(nn.Module):
    """
    PRNet model from 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network,'
    https://arxiv.org/abs/1803.07835.

    Parameters
    ----------
    channels : list(list(list(int)))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (256, 256)
        Spatial size of the expected input image.
    num_classes : int, default 3
        Number of classification classes.
    """
    def __init__(self,
                 channels: list[list[list[int]]],
                 init_block_channels: int,
                 bn_eps: float = 1e-5,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (256, 256),
                 num_classes: int = 3):
        super(PRNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        normalization = lambda_batchnorm2d(eps=bn_eps)

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv4x4_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            normalization=normalization))
        in_channels = init_block_channels

        encoder = nn.Sequential()
        for i, channels_per_stage in enumerate(channels[0]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) else 1
                padding = (1, 2, 1, 2) if (stride == 1) else 1
                stage.add_module("unit{}".format(j + 1), PRResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding,
                    normalization=normalization))
                in_channels = out_channels
            encoder.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("encoder", encoder)

        decoder = nn.Sequential()
        for i, channels_per_stage in enumerate(channels[1]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                padding = 3 if (stride == 1) else 1
                ext_padding = (2, 1, 2, 1) if (stride == 1) else None
                stage.add_module("unit{}".format(j + 1), deconv4x4_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding,
                    ext_padding=ext_padding,
                    normalization=normalization))
                in_channels = out_channels
            decoder.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("decoder", decoder)

        self.output = PROutputBlock(
            in_channels=in_channels,
            out_channels=num_classes,
            normalization=normalization)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_prnet(model_name: str | None = None,
              pretrained: bool = False,
              root: str = os.path.join("~", ".torch", "models"),
              **kwargs) -> nn.Module:
    """
    Create PRNet model with specific parameters.

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
    init_block_channels = 16
    enc_channels = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]
    dec_channels = [[512], [256, 256, 256], [128, 128, 128], [64, 64, 64], [32, 32], [16, 16]]
    channels = [enc_channels, dec_channels]

    net = PRNet(
        channels=channels,
        init_block_channels=init_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .common.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def prnet(**kwargs) -> nn.Module:
    """
    PRNet model for AFLW2000-3D from 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression
    Network,' https://arxiv.org/abs/1803.07835.

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
    return get_prnet(
        model_name="prnet",
        bn_eps=1e-3,
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        prnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != prnet or weight_count == 13353618)

        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 3, 256, 256))


if __name__ == "__main__":
    _test()
