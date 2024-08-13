"""
    SQNet for image segmentation, implemented in PyTorch.
    Original paper: 'Speeding up Semantic Segmentation for Autonomous Driving,'
    https://openreview.net/pdf?id=S1uHiFyyg.
"""

__all__ = ['SQNet', 'sqnet_cityscapes']

import os
import torch
import torch.nn as nn
from typing import Callable
from .common.norm import lambda_batchnorm2d
from .common.conv import conv1x1_block, conv3x3_block, deconv3x3_block
from .common.arch import Concurrent, Hourglass


class FireBlock(nn.Module):
    """
    SQNet specific encoder block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    normalization : function
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool,
                 normalization: Callable[..., nn.Module],
                 activation: Callable[..., nn.Module]):
        super(FireBlock, self).__init__()
        squeeze_channels = out_channels // 8
        expand_channels = out_channels // 2

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            bias=bias,
            normalization=normalization,
            activation=activation)
        self.branches = Concurrent(merge_type="cat")
        self.branches.add_module("branch1", conv1x1_block(
            in_channels=squeeze_channels,
            out_channels=expand_channels,
            bias=bias,
            normalization=normalization,
            activation=None))
        self.branches.add_module("branch2", conv3x3_block(
            in_channels=squeeze_channels,
            out_channels=expand_channels,
            bias=bias,
            normalization=normalization,
            activation=None))
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.branches(x)
        x = self.activ(x)
        return x


class ParallelDilatedConv(nn.Module):
    """
    SQNet specific decoder block (parallel dilated convolution).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    normalization : function
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool,
                 normalization: Callable[..., nn.Module],
                 activation: Callable[..., nn.Module]):
        super(ParallelDilatedConv, self).__init__()
        dilations = [1, 2, 3, 4]

        self.branches = Concurrent(merge_type="sum")
        for i, dilation in enumerate(dilations):
            self.branches.add_module("branch{}".format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=dilation,
                dilation=dilation,
                bias=bias,
                normalization=normalization,
                activation=activation))

    def forward(self, x):
        x = self.branches(x)
        return x


class SQNetUpStage(nn.Module):
    """
    SQNet upscale stage.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    normalization : function
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    use_parallel_conv : bool
        Whether to use parallel dilated convolution.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool,
                 normalization: Callable[..., nn.Module],
                 activation: Callable[..., nn.Module],
                 use_parallel_conv: bool):
        super(SQNetUpStage, self).__init__()

        if use_parallel_conv:
            self.conv = ParallelDilatedConv(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=bias,
                normalization=normalization,
                activation=activation)
        else:
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=bias,
                normalization=normalization,
                activation=activation)
        self.deconv = deconv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            bias=bias,
            normalization=normalization,
            activation=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x


class SQNet(nn.Module):
    """
    SQNet model from 'Speeding up Semantic Segmentation for Autonomous Driving,'
    https://openreview.net/pdf?id=S1uHiFyyg.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each stage in encoder and decoder.
    init_block_channels : int
        Number of output channels for the initial unit.
    layers : list(int)
        Number of layers for each stage in encoder.
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
                 channels: list[list[int]],
                 init_block_channels: int,
                 layers: list[int],
                 aux: bool = False,
                 fixed_size: bool = False,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (1024, 2048),
                 num_classes: int = 19):
        super(SQNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        bias = True
        use_bn = False
        normalization = lambda_batchnorm2d() if use_bn else None
        activation = (lambda: nn.ELU(inplace=True))

        self.stem = conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bias=bias,
            normalization=normalization,
            activation=activation)
        in_channels = init_block_channels

        down_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        for i, out_channels in enumerate(channels[0]):
            skip_seq.add_module("skip{}".format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=bias,
                normalization=normalization,
                activation=activation))
            stage = nn.Sequential()
            stage.add_module("unit1", nn.MaxPool2d(
                kernel_size=2,
                stride=2))
            for j in range(layers[i]):
                stage.add_module("unit{}".format(j + 2), FireBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=bias,
                    normalization=normalization,
                    activation=activation))
                in_channels = out_channels
            down_seq.add_module("down{}".format(i + 1), stage)

        in_channels = in_channels // 2

        up_seq = nn.Sequential()
        for i, out_channels in enumerate(channels[1]):
            use_parallel_conv = True if i == 0 else False
            up_seq.add_module("up{}".format(i + 1), SQNetUpStage(
                in_channels=(2 * in_channels),
                out_channels=out_channels,
                bias=bias,
                normalization=normalization,
                activation=activation,
                use_parallel_conv=use_parallel_conv))
            in_channels = out_channels
        up_seq = up_seq[::-1]

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            merge_type="cat")

        self.head = SQNetUpStage(
            in_channels=(2 * in_channels),
            out_channels=num_classes,
            bias=bias,
            normalization=normalization,
            activation=activation,
            use_parallel_conv=False)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.hg(x)
        x = self.head(x)
        return x


def get_sqnet(model_name: str | None = None,
              pretrained: bool = False,
              root: str = os.path.join("~", ".torch", "models"),
              **kwargs) -> nn.Module:
    """
    Create SQNet model with specific parameters.

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
    channels = [[128, 256, 512], [256, 128, 96]]
    init_block_channels = 96
    layers = [2, 2, 3]

    net = SQNet(
        channels=channels,
        init_block_channels=init_block_channels,
        layers=layers,
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


def sqnet_cityscapes(num_classes: int = 19,
                     **kwargs) -> nn.Module:
    """
    SQNet model for Cityscapes from 'Speeding up Semantic Segmentation for Autonomous Driving,'
    https://openreview.net/pdf?id=S1uHiFyyg.

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
    return get_sqnet(
        num_classes=num_classes,
        model_name="sqnet_cityscapes",
        **kwargs)


def _test():
    from .common.model_store import calc_net_weight_count

    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        sqnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sqnet_cityscapes or weight_count == 16262771)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
