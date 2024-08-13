"""
    InceptionResNetV2 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionResNetV2', 'inceptionresnetv2']

import os
import torch.nn as nn
from typing import Callable
from .common.norm import lambda_batchnorm2d
from .common.conv import conv1x1_block, conv3x3_block
from .common.arch import Concurrent
from .inceptionv3 import AvgPoolBranch, Conv1x1Branch, ConvSeqBranch
from .inceptionresnetv1 import InceptionAUnit, InceptionBUnit, InceptionCUnit, ReductionAUnit, ReductionBUnit


class InceptBlock5b(nn.Module):
    """
    InceptionResNetV2 type Mixed-5b block.

    Parameters
    ----------
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 normalization: Callable[..., nn.Module]):
        super(InceptBlock5b, self).__init__()
        in_channels = 192

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=96,
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
            out_channels=64,
            normalization=normalization,
            count_include_pad=False))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(nn.Module):
    """
    InceptionResNetV2 specific initial block.

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
        self.block = InceptBlock5b(normalization=normalization)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.block(x)
        return x


class InceptionResNetV2(nn.Module):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
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
        super(InceptionResNetV2, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        normalization = lambda_batchnorm2d(eps=bn_eps)
        layers = [10, 21, 11]
        in_channels_list = [320, 1088, 2080]
        normal_out_channels_list = [[32, 32, 32, 32, 48, 64], [192, 128, 160, 192], [192, 192, 224, 256]]
        reduction_out_channels_list = [[384, 256, 256, 384], [256, 384, 256, 288, 256, 288, 320]]

        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = nn.Sequential()
        self.features.add_module("init_block", InceptInitBlock(
            in_channels=in_channels,
            normalization=normalization))
        in_channels = in_channels_list[0]
        for i, layers_per_stage in enumerate(layers):
            stage = nn.Sequential()
            for j in range(layers_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                    out_channels_list_per_stage = reduction_out_channels_list[i - 1]
                else:
                    unit = normal_units[i]
                    out_channels_list_per_stage = normal_out_channels_list[i]
                if (i == len(layers) - 1) and (j == layers_per_stage - 1):
                    unit_kwargs = {"scale": 1.0, "activate": False}
                else:
                    unit_kwargs = {}
                stage.add_module("unit{}".format(j + 1), unit(
                    in_channels=in_channels,
                    out_channels_list=out_channels_list_per_stage,
                    normalization=normalization,
                    **unit_kwargs))
                if (j == 0) and (i != 0):
                    in_channels = in_channels_list[i]
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_conv", conv1x1_block(
            in_channels=in_channels,
            out_channels=1536,
            normalization=normalization))
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


def get_inceptionresnetv2(model_name: str | None = None,
                          pretrained: bool = False,
                          root: str = os.path.join("~", ".torch", "models"),
                          **kwargs) -> nn.Module:
    """
    Create InceptionResNetV2 model with specific parameters.

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
    net = InceptionResNetV2(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .common.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def inceptionresnetv2(**kwargs) -> nn.Module:
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
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
    return get_inceptionresnetv2(
        model_name="inceptionresnetv2",
        bn_eps=1e-3,
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        inceptionresnetv2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionresnetv2 or weight_count == 55843464)

        x = torch.randn(1, 3, 299, 299)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
