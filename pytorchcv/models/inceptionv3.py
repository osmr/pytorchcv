"""
    InceptionV3 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.
"""

__all__ = ['InceptionV3', 'inceptionv3', 'MaxPoolBranch', 'AvgPoolBranch', 'Conv1x1Branch', 'ConvSeqBranch']

import os
import torch
import torch.nn as nn
from typing import Callable
from .common.activ import lambda_relu
from .common.norm import lambda_batchnorm2d
from .common.conv import ConvBlock, conv1x1_block, conv3x3_block
from .common.arch import Concurrent


class MaxPoolBranch(nn.Module):
    """
    Inception specific max pooling branch block.
    """
    def __init__(self):
        super(MaxPoolBranch, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.pool(x)
        return x


class AvgPoolBranch(nn.Module):
    """
    Inception specific average pooling branch block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    normalization : function
        Lambda-function generator for normalization layer.
    count_include_pad : bool, default True
        Whether to include the zero-padding in the averaging calculation.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 normalization: Callable[..., nn.Module],
                 count_include_pad: bool = True):
        super(AvgPoolBranch, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=count_include_pad)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            normalization=normalization)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Conv1x1Branch(nn.Module):
    """
    Inception specific convolutional 1x1 branch block.

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
        super(Conv1x1Branch, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            normalization=normalization)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvSeqBranch(nn.Module):
    """
    Inception specific convolutional sequence branch block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list(int) or tuple(int, ...)
        List of numbers of output channels.
    kernel_size_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of convolution window sizes.
    strides_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of strides of the convolution.
    padding_list : list(int) or tuple(int, ...) or tuple(int or tuple(int, int), ...)
        List of padding values for convolution layers.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or None, default lambda_relu()
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels_list: list[int] | tuple[int, ...],
                 kernel_size_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 strides_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 padding_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | None = lambda_relu()):
        super(ConvSeqBranch, self).__init__()
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                bias=bias,
                normalization=normalization))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv_list(x)
        return x


class ConvSeq3x3Branch(nn.Module):
    """
    InceptionV3 specific convolutional sequence branch block with splitting by 3x3.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list(int) or tuple(int, ...)
        List of numbers of output channels.
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
                 out_channels_list: list[int] | tuple[int, ...],
                 kernel_size_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 strides_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 padding_list: list[int] | tuple[int, ...] | tuple[int | tuple[int, int], ...],
                 normalization: Callable[..., nn.Module]):
        super(ConvSeq3x3Branch, self).__init__()
        self.conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                normalization=normalization))
            in_channels = out_channels
        self.conv1x3 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            normalization=normalization)
        self.conv3x1 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
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
    InceptionV3 type Inception-A unit.

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
        super(InceptionAUnit, self).__init__()
        assert (out_channels > 224)
        pool_out_channels = out_channels - 224

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=64,
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(48, 64),
            kernel_size_list=(1, 5),
            strides_list=(1, 1),
            padding_list=(0, 2),
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
            out_channels=pool_out_channels,
            normalization=normalization))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionAUnit(nn.Module):
    """
    InceptionV3 type Reduction-A unit.

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
        super(ReductionAUnit, self).__init__()
        assert (in_channels == 288)
        assert (out_channels == 768)

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
            out_channels_list=(64, 96, 96),
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
    InceptionV3 type Inception-B unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of output channels in the 7x7 branches.
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 normalization: Callable[..., nn.Module]):
        super(InceptionBUnit, self).__init__()
        assert (in_channels == 768)
        assert (out_channels == 768)

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=192,
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(mid_channels, mid_channels, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            normalization=normalization))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(mid_channels, mid_channels, mid_channels, mid_channels, 192),
            kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
            strides_list=(1, 1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3)),
            normalization=normalization))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=192,
            normalization=normalization))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionBUnit(nn.Module):
    """
    InceptionV3 type Reduction-B unit.

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
        super(ReductionBUnit, self).__init__()
        assert (in_channels == 768)
        assert (out_channels == 1280)

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 320),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192, 192, 192),
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
    InceptionV3 type Inception-C unit.

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
        super(InceptionCUnit, self).__init__()
        assert (out_channels == 2048)

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=320,
            normalization=normalization))
        self.branches.add_module("branch2", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(1,),
            strides_list=(1,),
            padding_list=(0,),
            normalization=normalization))
        self.branches.add_module("branch3", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels_list=(448, 384),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            normalization=normalization))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=192,
            normalization=normalization))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(nn.Module):
    """
    InceptionV3 specific initial block.

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
        super(InceptInitBlock, self).__init__()
        assert (out_channels == 192)

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
        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)
        self.conv4 = conv1x1_block(
            in_channels=64,
            out_channels=80,
            stride=1,
            padding=0,
            normalization=normalization)
        self.conv5 = conv3x3_block(
            in_channels=80,
            out_channels=192,
            stride=1,
            padding=0,
            normalization=normalization)
        self.pool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        return x


class InceptionV3(nn.Module):
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    b_mid_channels : list(int)
        Number of middle channels for each Inception-B unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (299, 299)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels: list[list[int]],
                 init_block_channels: int,
                 b_mid_channels: list[int],
                 bn_eps: float = 1e-5,
                 dropout_rate: float = 0.5,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (299, 299),
                 num_classes: int = 1000):
        super(InceptionV3, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        normalization = lambda_batchnorm2d(eps=bn_eps)
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = nn.Sequential()
        self.features.add_module("init_block", InceptInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            normalization=normalization))
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                else:
                    unit = normal_units[i]
                if unit == InceptionBUnit:
                    stage.add_module("unit{}".format(j + 1), unit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        mid_channels=b_mid_channels[j - 1],
                        normalization=normalization))
                else:
                    stage.add_module("unit{}".format(j + 1), unit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        normalization=normalization))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self.output = nn.Sequential()
        self.output.add_module("dropout", nn.Dropout(p=dropout_rate))
        self.output.add_module("fc", nn.Linear(
            in_features=in_channels,
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


def get_inceptionv3(model_name: str | None = None,
                    pretrained: bool = False,
                    root: str = os.path.join("~", ".torch", "models"),
                    **kwargs) -> nn.Module:
    """
    Create InceptionV3 model with specific parameters.

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
    init_block_channels = 192
    channels = [[256, 288, 288],
                [768, 768, 768, 768, 768],
                [1280, 2048, 2048]]
    b_mid_channels = [128, 160, 160, 192]

    net = InceptionV3(
        channels=channels,
        init_block_channels=init_block_channels,
        b_mid_channels=b_mid_channels,
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


def inceptionv3(**kwargs) -> nn.Module:
    """
    InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

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
    return get_inceptionv3(
        model_name="inceptionv3",
        bn_eps=1e-3,
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        inceptionv3,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionv3 or weight_count == 23834568)

        x = torch.randn(1, 3, 299, 299)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
