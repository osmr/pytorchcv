"""
    InceptionV4 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionV4', 'inceptionv4']

import os
import torch
import torch.nn as nn
from typing import Callable
from .common.norm import lambda_batchnorm2d
from .common.conv import ConvBlock, conv3x3_block
from .common.arch import Concurrent
from .inceptionv3 import MaxPoolBranch, AvgPoolBranch, Conv1x1Branch, ConvSeqBranch


class Conv3x3Branch(nn.Module):
    """
    InceptionV4 specific convolutional 3x3 branch block.

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
        super(Conv3x3Branch, self).__init__()
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            padding=0,
            normalization=normalization)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvSeq3x3Branch(nn.Module):
    """
    InceptionV4 specific convolutional sequence branch block with splitting by 3x3.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels_list : list(int) or tuple(int, ...)
        List of numbers of output channels for middle layers.
    kernel_size_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of convolution window sizes.
    strides_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of strides of the convolution.
    padding_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of padding values for convolution layers.
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels_list: list[int] | tuple[int, ...],
                 kernel_size_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 strides_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 padding_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 normalization: Callable[..., nn.Module]):
        super(ConvSeq3x3Branch, self).__init__()
        self.conv_list = nn.Sequential()
        for i, (mid_channels, kernel_size, strides, padding) in enumerate(zip(
                mid_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                normalization=normalization))
            in_channels = mid_channels
        self.conv1x3 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            normalization=normalization)
        self.conv3x1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            normalization=normalization)

    def forward(self, x):
        x = self.conv_list(x)
        y1 = self.conv1x3(x)
        y2 = self.conv3x1(x)
        x = torch.cat((y1, y2), dim=1)
        return x


class InceptionAUnit(nn.Module):
    """
    InceptionV4 type Inception-A unit.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(InceptionAUnit, self).__init__()
        in_channels = 384

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=96,
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            normalization=normalization))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            normalization=normalization))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=96,
            normalization=normalization,
            count_include_pad=False))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionAUnit(nn.Module):
    """
    InceptionV4 type Reduction-A unit.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(ReductionAUnit, self).__init__()
        in_channels = 384

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,),
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            normalization=normalization))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptionBUnit(nn.Module):
    """
    InceptionV4 type Inception-B unit.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(InceptionBUnit, self).__init__()
        in_channels = 1024

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=384,
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            normalization=normalization))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192, 224, 224, 256),
            kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
            strides_list=(1, 1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3)),
            normalization=normalization))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=128,
            normalization=normalization,
            count_include_pad=False))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionBUnit(nn.Module):
    """
    InceptionV4 type Reduction-B unit.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(ReductionBUnit, self).__init__()
        in_channels = 1024

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256, 320, 320),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 2),
            padding_list=(0, (0, 3), (3, 0), 0),
            normalization=normalization))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptionCUnit(nn.Module):
    """
    InceptionV4 type Inception-C unit.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(InceptionCUnit, self).__init__()
        in_channels = 1536

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=256,
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels=256,
            mid_channels_list=(384,),
            kernel_size_list=(1,),
            strides_list=(1,),
            padding_list=(0,),
            normalization=normalization))
        self.branches.add_module("branch3", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels=256,
            mid_channels_list=(384, 448, 512),
            kernel_size_list=(1, (3, 1), (1, 3)),
            strides_list=(1, 1, 1),
            padding_list=(0, (1, 0), (0, 1)),
            normalization=normalization))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=256,
            normalization=normalization,
            count_include_pad=False))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptBlock3a(nn.Module):
    """
    InceptionV4 type Mixed-3a block.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(InceptBlock3a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", MaxPoolBranch())
        self.branches.add_module("branch2", Conv3x3Branch(
            in_channels=64,
            out_channels=96,
            normalization=normalization))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptBlock4a(nn.Module):
    """
    InceptionV4 type Mixed-4a block.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(InceptBlock4a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 0),
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 64, 64, 96),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 1),
            padding_list=(0, (0, 3), (3, 0), 0),
            normalization=normalization))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptBlock5a(nn.Module):
    """
    InceptionV4 type Mixed-5a block.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(InceptBlock5a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv3x3Branch(
            in_channels=192,
            out_channels=192,
            normalization=normalization))
        self.branches.add_module("branch2", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(nn.Module):
    """
    InceptionV4 specific initial block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 normalization: Callable[..., nn.Module]):
        super(InceptInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=32,
            stride=2,
            padding=0,
            normalization=normalization)
        self.conv2 = conv3x3_block(
            in_channels=32,
            out_channels=32,
            stride=1,
            padding=0,
            normalization=normalization)
        self.conv3 = conv3x3_block(
            in_channels=32,
            out_channels=64,
            stride=1,
            padding=1,
            normalization=normalization)
        self.block1 = InceptBlock3a(normalization=normalization)
        self.block2 = InceptBlock4a(normalization=normalization)
        self.block3 = InceptBlock5a(normalization=normalization)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class InceptionV4(nn.Module):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters
    ----------
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (299, 299)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 dropout_rate: float = 0.0,
                 bn_eps: float = 1e-5,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (299, 299),
                 num_classes: int = 1000):
        super(InceptionV4, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        normalization = lambda_batchnorm2d(eps=bn_eps)
        layers = [4, 8, 4]
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = nn.Sequential()
        self.features.add_module("init_block", InceptInitBlock(
            in_channels=in_channels,
            normalization=normalization))

        for i, layers_per_stage in enumerate(layers):
            stage = nn.Sequential()
            for j in range(layers_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                else:
                    unit = normal_units[i]
                stage.add_module("unit{}".format(j + 1), unit(normalization=normalization))
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self.output = nn.Sequential()
        if dropout_rate > 0.0:
            self.output.add_module("dropout", nn.Dropout(p=dropout_rate))
        self.output.add_module("fc", nn.Linear(
            in_features=1536,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_inceptionv4(model_name: str | None = None,
                    pretrained: bool = False,
                    root: str = os.path.join("~", ".torch", "models"),
                    **kwargs) -> nn.Module:
    """
    Create InceptionV4 model with specific parameters.

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
    net = InceptionV4(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .common.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def inceptionv4(**kwargs) -> nn.Module:
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

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
    return get_inceptionv4(
        model_name="inceptionv4",
        bn_eps=1e-3,
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        inceptionv4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionv4 or weight_count == 42679816)

        x = torch.randn(1, 3, 299, 299)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
