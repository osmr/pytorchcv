"""
    WRN for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.
"""

__all__ = ['WRN', 'wrn50_2']

import os
import torch.nn as nn


class WRNConv(nn.Module):
    """
    WRN specific convolution block.

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
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 activate: bool):
        super(WRNConv, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            x = self.activ(x)
        return x


def wrn_conv1x1(in_channels: int,
                out_channels: int,
                stride: int | tuple[int, int],
                activate: bool):
    """
    1x1 version of the WRN specific convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return WRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        activate=activate)


def wrn_conv3x3(in_channels: int,
                out_channels: int,
                stride: int | tuple[int, int],
                activate: bool):
    """
    3x3 version of the WRN specific convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return WRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        activate=activate)


class WRNBottleneck(nn.Module):
    """
    WRN bottleneck block for residual path in WRN unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 width_factor: float):
        super(WRNBottleneck, self).__init__()
        mid_channels = int(round(out_channels // 4 * width_factor))

        self.conv1 = wrn_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=1,
            activate=True)
        self.conv2 = wrn_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activate=True)
        self.conv3 = wrn_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=1,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class WRNUnit(nn.Module):
    """
    WRN unit with residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 width_factor: float):
        super(WRNUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = WRNBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            width_factor=width_factor)
        if self.resize_identity:
            self.identity_conv = wrn_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activate=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class WRNInitBlock(nn.Module):
    """
    WRN specific initial block.

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
        super(WRNInitBlock, self).__init__()
        self.conv = WRNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            activate=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class WRN(nn.Module):
    """
    WRN model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    width_factor : float
        Wide scale factor for width of layers.
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
                 width_factor: float,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (224, 224),
                 num_classes: int = 1000):
        super(WRN, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", WRNInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), WRNUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    width_factor=width_factor))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

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


def get_wrn(blocks: int,
            width_factor: float,
            model_name: str | None = None,
            pretrained: bool = False,
            root: str = os.path.join("~", ".torch", "models"),
            **kwargs) -> nn.Module:
    """
    Create WRN model with specific parameters.

    Parameters
    ----------
    blocks : int
        Number of blocks.
    width_factor : float
        Wide scale factor for width of layers.
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
    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported WRN with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = WRN(
        channels=channels,
        init_block_channels=init_block_channels,
        width_factor=width_factor,
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


def wrn50_2(**kwargs) -> nn.Module:
    """
    WRN-50-2 model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

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
    return get_wrn(
        blocks=50,
        width_factor=2.0,
        model_name="wrn50_2",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        wrn50_2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn50_2 or weight_count == 68849128)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
