"""
    CondenseNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.
"""

__all__ = ['CondenseNet', 'condensenet74_c4_g4', 'condensenet74_c8_g8']

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from .common.tutti import ChannelShuffle


class CondenseSimpleConv(nn.Module):
    """
    CondenseNet specific simple convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int)
        Padding value for convolution layer.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 groups: int):
        super(CondenseSimpleConv, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


def condense_simple_conv3x3(in_channels: int,
                            out_channels: int,
                            groups: int):
    """
    3x3 version of the CondenseNet specific simple convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    """
    return CondenseSimpleConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=groups)


class CondenseComplexConv(nn.Module):
    """
    CondenseNet specific complex convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int)
        Padding value for convolution layer.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 groups: int):
        super(CondenseComplexConv, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.c_shuffle = ChannelShuffle(
            channels=out_channels,
            groups=groups)
        self.register_buffer("index", torch.LongTensor(in_channels))
        self.index.fill_(0)

    def forward(self, x):
        x = torch.index_select(x, dim=1, index=Variable(self.index))
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        x = self.c_shuffle(x)
        return x


def condense_complex_conv1x1(in_channels: int,
                             out_channels: int,
                             groups: int):
    """
    1x1 version of the CondenseNet specific complex convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    """
    return CondenseComplexConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=groups)


class CondenseUnit(nn.Module):
    """
    CondenseNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int):
        super(CondenseUnit, self).__init__()
        bottleneck_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bottleneck_size

        self.conv1 = condense_complex_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=groups)
        self.conv2 = condense_simple_conv3x3(
            in_channels=mid_channels,
            out_channels=inc_channels,
            groups=groups)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat((identity, x), dim=1)
        return x


class TransitionBlock(nn.Module):
    """
    CondenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.
    """
    def __init__(self):
        super(TransitionBlock, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseInitBlock(nn.Module):
    """
    CondenseNet specific initial block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(CondenseInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class PostActivation(nn.Module):
    """
    CondenseNet final block, which performs the same function of postactivation as in PreResNet.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels: int):
        super(PostActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class CondenseLinear(nn.Module):
    """
    CondenseNet specific linear block.

    Parameters
    ----------
    in_features : int
        Number of input channels.
    out_features : int
        Number of output channels.
    drop_rate : float, default 0.5
        Fraction of input channels for drop.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 drop_rate: float = 0.5):
        super(CondenseLinear, self).__init__()
        drop_in_features = int(in_features * drop_rate)
        self.linear = nn.Linear(
            in_features=drop_in_features,
            out_features=out_features)
        self.register_buffer("index", torch.LongTensor(drop_in_features))
        self.index.fill_(0)

    def forward(self, x):
        x = torch.index_select(x, dim=1, index=Variable(self.index))
        x = self.linear(x)
        return x


class CondenseNet(nn.Module):
    """
    CondenseNet model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    groups : int
        Number of groups in convolution layers.
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
                 groups: int,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (224, 224),
                 num_classes: int = 1000):
        super(CondenseNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", CondenseInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            if i != 0:
                stage.add_module("trans{}".format(i + 1), TransitionBlock())
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), CondenseUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("post_activ", PostActivation(in_channels=in_channels))
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = CondenseLinear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_condensenet(num_layers: int,
                    groups: int = 4,
                    model_name: str | None = None,
                    pretrained: bool = False,
                    root: str = os.path.join("~", ".torch", "models"),
                    **kwargs) -> nn.Module:
    """
    Create CondenseNet (converted) model with specific parameters.

    Parameters
    ----------
    num_layers : int
        Number of layers.
    groups : int, default 4
        Number of groups in convolution layers.
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
    if num_layers == 74:
        init_block_channels = 16
        layers = [4, 6, 8, 10, 8]
        growth_rates = [8, 16, 32, 64, 128]
    else:
        raise ValueError("Unsupported CondenseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(lambda xi, yi:
                      xi + [reduce(lambda xj, yj:
                                   xj + [xj[-1] + yj],
                                   [yi[1]] * yi[0],
                                   [xi[-1][-1]])[1:]],
                      zip(layers, growth_rates),
                      [[init_block_channels]])[1:]

    net = CondenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        groups=groups,
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


def condensenet74_c4_g4(**kwargs) -> nn.Module:
    """
    CondenseNet-74 (C=G=4) model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

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
    return get_condensenet(
        num_layers=74,
        groups=4,
        model_name="condensenet74_c4_g4",
        **kwargs)


def condensenet74_c8_g8(**kwargs) -> nn.Module:
    """
    CondenseNet-74 (C=G=8) model (converted) from 'CondenseNet: An Efficient DenseNet using Learned Group Convolutions,'
    https://arxiv.org/abs/1711.09224.

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
    return get_condensenet(
        num_layers=74,
        groups=8,
        model_name="condensenet74_c8_g8",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        condensenet74_c4_g4,
        condensenet74_c8_g8,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != condensenet74_c4_g4 or weight_count == 4773944)
        assert (model != condensenet74_c8_g8 or weight_count == 2935416)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
