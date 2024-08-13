"""
    ESPNet-C for image segmentation, implemented in PyTorch.
    Original paper: 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation,'
    https://arxiv.org/abs/1803.06815.
"""

__all__ = ['ESPCNet', 'espcnet_cityscapes', 'ESPBlock']

import os
import torch
import torch.nn as nn
from typing import Callable
from .common.activ import lambda_prelu
from .common.norm import lambda_batchnorm2d
from .common.conv import conv1x1, conv3x3, conv3x3_block
from .common.arch import DualPathSequential
from .common.tutti import NormActivation, InterpolationBlock


class HierarchicalConcurrent(nn.Sequential):
    """
    A container for hierarchical concatenation of modules on the base of the sequential container.

    Parameters
    ----------
    exclude_first : bool, default False
        Whether to exclude the first branch in the intermediate sum.
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self,
                 exclude_first: bool = False,
                 axis: int = 1):
        super(HierarchicalConcurrent, self).__init__()
        self.exclude_first = exclude_first
        self.axis = axis

    def forward(self, x):
        out = []
        y_prev = None
        for i, module in enumerate(self._modules.values()):
            y = module(x)
            if y_prev is not None:
                y += y_prev
            out.append(y)
            if (not self.exclude_first) or (i > 0):
                y_prev = y
        out = torch.cat(tuple(out), dim=self.axis)
        return out


class ESPBlock(nn.Module):
    """
    ESPNet block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    downsample : bool
        Whether to downsample image.
    residual : bool
        Whether to use residual connection.
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool,
                 residual: bool,
                 normalization: Callable[..., nn.Module]):
        super(ESPBlock, self).__init__()
        self.residual = residual
        dilations = [1, 2, 4, 8, 16]
        num_branches = len(dilations)
        mid_channels = out_channels // num_branches
        extra_mid_channels = out_channels - (num_branches - 1) * mid_channels

        if downsample:
            self.reduce_conv = conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=2)
        else:
            self.reduce_conv = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)

        self.branches = HierarchicalConcurrent(exclude_first=True)
        for i in range(num_branches):
            out_channels_i = extra_mid_channels if i == 0 else mid_channels
            self.branches.add_module("branch{}".format(i + 1), conv3x3(
                in_channels=mid_channels,
                out_channels=out_channels_i,
                padding=dilations[i],
                dilation=dilations[i]))

        self.norm_activ = NormActivation(
            in_channels=out_channels,
            normalization=normalization,
            activation=lambda_prelu(num_parameters=out_channels))

    def forward(self, x):
        y = self.reduce_conv(x)
        y = self.branches(y)
        if self.residual:
            y = y + x
        y = self.norm_activ(y)
        return y


class ESPUnit(nn.Module):
    """
    ESPNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    layers : int
        Number of layers.
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers: int,
                 normalization: Callable[..., nn.Module]):
        super(ESPUnit, self).__init__()
        mid_channels = out_channels // 2

        self.down = ESPBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            downsample=True,
            residual=False,
            normalization=normalization)
        self.blocks = nn.Sequential()
        for i in range(layers - 1):
            self.blocks.add_module("block{}".format(i + 1), ESPBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                downsample=False,
                residual=True,
                normalization=normalization))

    def forward(self, x):
        x = self.down(x)
        y = self.blocks(x)
        x = torch.cat((y, x), dim=1)  # NB: This differs from the original implementation.
        return x


class ESPStage(nn.Module):
    """
    ESPNet stage.

    Parameters
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    layers : int
        Number of layers in the unit.
    normalization : function
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 x_channels: int,
                 y_in_channels: int,
                 y_out_channels: int,
                 layers: int,
                 normalization: Callable[..., nn.Module]):
        super(ESPStage, self).__init__()
        self.use_x = (x_channels > 0)
        self.use_unit = (layers > 0)

        if self.use_x:
            self.x_down = nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1)

        if self.use_unit:
            self.unit = ESPUnit(
                in_channels=y_in_channels,
                out_channels=(y_out_channels - x_channels),
                layers=layers,
                normalization=normalization)

        self.norm_activ = NormActivation(
            in_channels=y_out_channels,
            normalization=normalization,
            activation=lambda_prelu(num_parameters=y_out_channels))

    def forward(self, y, x=None):
        if self.use_unit:
            y = self.unit(y)
        if self.use_x:
            x = self.x_down(x)
            y = torch.cat((y, x), dim=1)
        y = self.norm_activ(y)
        return y, x


class ESPCNet(nn.Module):
    """
    ESPNet-C model from 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation,'
    https://arxiv.org/abs/1803.06815.

    Parameters
    ----------
    layers : list(int)
        Number of layers for each unit.
    channels : list(int)
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    cut_x : list(int)
        Whether to concatenate with x-branch for each unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (1024, 2048)
        Spatial size of the expected input image.
    num_classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 layers: list[int],
                 channels: list[int],
                 init_block_channels: int,
                 cut_x: list[int],
                 bn_eps: bool = 1e-5,
                 aux: bool = False,
                 fixed_size: bool = False,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (1024, 2048),
                 num_classes: int = 19):
        super(ESPCNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        normalization = lambda_batchnorm2d(bn_eps)

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=0)
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            normalization=normalization,
            activation=lambda_prelu(num_parameters=init_block_channels)))
        y_in_channels = init_block_channels

        for i, (layers_i, y_out_channels) in enumerate(zip(layers, channels)):
            self.features.add_module("stage{}".format(i + 1), ESPStage(
                x_channels=in_channels if cut_x[i] == 1 else 0,
                y_in_channels=y_in_channels,
                y_out_channels=y_out_channels,
                layers=layers_i,
                normalization=normalization))
            y_in_channels = y_out_channels

        self.head = conv1x1(
            in_channels=y_in_channels,
            out_channels=num_classes)

        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=False)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        y = self.features(x, x)
        y = self.head(y)
        y = self.up(y, size=in_size)
        return y


def get_espcnet(model_name: str | None = None,
                pretrained: bool = False,
                root: str = os.path.join("~", ".torch", "models"),
                **kwargs) -> nn.Module:
    """
    Create ESPNet-C model with specific parameters.

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
    layers = [0, 6, 4]
    channels = [19, 131, 256]
    cut_x = [1, 1, 0]
    bn_eps = 1e-3

    net = ESPCNet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        cut_x=cut_x,
        bn_eps=bn_eps,
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


def espcnet_cityscapes(num_classes: int = 19,
                       **kwargs) -> nn.Module:
    """
    ESPNet-C model for Cityscapes from 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic
    Segmentation,' https://arxiv.org/abs/1803.06815.

    Parameters
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_espcnet(
        num_classes=num_classes,
        model_name="espcnet_cityscapes",
        **kwargs)


def _test():
    from .common.model_store import calc_net_weight_count

    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        espcnet_cityscapes,
    ]

    for model in models:

        net = model(
            pretrained=pretrained,
            in_size=in_size,
            fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != espcnet_cityscapes or weight_count == 210889)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
