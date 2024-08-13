"""
    SqueezeNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size,'
    https://arxiv.org/abs/1602.07360.
"""

__all__ = ['SqueezeNet', 'squeezenet_v1_0', 'squeezenet_v1_1', 'squeezeresnet_v1_0', 'squeezeresnet_v1_1']

import os
import torch
import torch.nn as nn


class FireConv(nn.Module):
    """
    SqueezeNet specific convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    padding : int or tuple(int, int)
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 padding: int | tuple[int, int]):
        super(FireConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class FireUnit(nn.Module):
    """
    SqueezeNet unit, so-called 'Fire' unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    squeeze_channels : int
        Number of output channels for squeeze convolution blocks.
    expand1x1_channels : int
        Number of output channels for expand 1x1 convolution blocks.
    expand3x3_channels : int
        Number of output channels for expand 3x3 convolution blocks.
    residual : bool
        Whether to use residual connection.
    """
    def __init__(self,
                 in_channels: int,
                 squeeze_channels: int,
                 expand1x1_channels: int,
                 expand3x3_channels: int,
                 residual: bool):
        super(FireUnit, self).__init__()
        self.residual = residual

        self.squeeze = FireConv(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            padding=0)
        self.expand1x1 = FireConv(
            in_channels=squeeze_channels,
            out_channels=expand1x1_channels,
            kernel_size=1,
            padding=0)
        self.expand3x3 = FireConv(
            in_channels=squeeze_channels,
            out_channels=expand3x3_channels,
            kernel_size=3,
            padding=1)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = torch.cat((y1, y2), dim=1)
        if self.residual:
            out = out + identity
        return out


class SqueezeInitBlock(nn.Module):
    """
    SqueezeNet specific initial block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple(int, int)
        Convolution window size.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int]):
        super(SqueezeInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class SqueezeNet(nn.Module):
    """
    SqueezeNet model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size,'
    https://arxiv.org/abs/1602.07360.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    residuals : list(list(int)) or None
        Whether to use residual units.
    init_block_kernel_size : int or tuple(int, int)
        The dimensions of the convolution window for the initial unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels: list[list[int]],
                 residuals: list[list[int]] | None,
                 init_block_kernel_size: int | tuple[int, int],
                 init_block_channels: int,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (224, 224),
                 num_classes: int = 1000):
        super(SqueezeNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", SqueezeInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=init_block_kernel_size))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            stage.add_module("pool{}".format(i + 1), nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True))
            for j, out_channels in enumerate(channels_per_stage):
                expand_channels = out_channels // 2
                squeeze_channels = out_channels // 8
                stage.add_module("unit{}".format(j + 1), FireUnit(
                    in_channels=in_channels,
                    squeeze_channels=squeeze_channels,
                    expand1x1_channels=expand_channels,
                    expand3x3_channels=expand_channels,
                    residual=((residuals is not None) and (residuals[i][j] == 1))))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("dropout", nn.Dropout(p=0.5))

        self.output = nn.Sequential()
        self.output.add_module("final_conv", nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1))
        self.output.add_module("final_activ", nn.ReLU(inplace=True))
        self.output.add_module("final_pool", nn.AvgPool2d(
            kernel_size=13,
            stride=1))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if "final_conv" in name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_squeezenet(version: str,
                   residual: bool = False,
                   model_name: str | None = None,
                   pretrained: bool = False,
                   root: str = os.path.join("~", ".torch", "models"),
                   **kwargs) -> nn.Module:
    """
    Create SqueezeNet model with specific parameters.

    Parameters
    ----------
    version : str
        Version of SqueezeNet ('1.0' or '1.1').
    residual : bool, default False
        Whether to use residual connections.
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
    if version == "1.0":
        channels = [[128, 128, 256], [256, 384, 384, 512], [512]]
        residuals = [[0, 1, 0], [1, 0, 1, 0], [1]]
        init_block_kernel_size = 7
        init_block_channels = 96
    elif version == "1.1":
        channels = [[128, 128], [256, 256], [384, 384, 512, 512]]
        residuals = [[0, 1], [0, 1], [0, 1, 0, 1]]
        init_block_kernel_size = 3
        init_block_channels = 64
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    if not residual:
        residuals = None

    net = SqueezeNet(
        channels=channels,
        residuals=residuals,
        init_block_kernel_size=init_block_kernel_size,
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


def squeezenet_v1_0(**kwargs) -> nn.Module:
    """
    SqueezeNet 'vanilla' model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
    size,' https://arxiv.org/abs/1602.07360.

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
    return get_squeezenet(
        version="1.0",
        residual=False,
        model_name="squeezenet_v1_0",
        **kwargs)


def squeezenet_v1_1(**kwargs) -> nn.Module:
    """
    SqueezeNet v1.1 model from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
    size,' https://arxiv.org/abs/1602.07360.

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
    return get_squeezenet(
        version="1.1",
        residual=False,
        model_name="squeezenet_v1_1",
        **kwargs)


def squeezeresnet_v1_0(**kwargs) -> nn.Module:
    """
    SqueezeNet model with residual connections from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
    <0.5MB model size,' https://arxiv.org/abs/1602.07360.

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
    return get_squeezenet(
        version="1.0",
        residual=True,
        model_name="squeezeresnet_v1_0",
        **kwargs)


def squeezeresnet_v1_1(**kwargs) -> nn.Module:
    """
    SqueezeNet v1.1 model with residual connections from 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size,' https://arxiv.org/abs/1602.07360.

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
    return get_squeezenet(
        version="1.1",
        residual=True,
        model_name="squeezeresnet_v1_1",
        **kwargs)


def _test():
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        squeezenet_v1_0,
        squeezenet_v1_1,
        squeezeresnet_v1_0,
        squeezeresnet_v1_1,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != squeezenet_v1_0 or weight_count == 1248424)
        assert (model != squeezenet_v1_1 or weight_count == 1235496)
        assert (model != squeezeresnet_v1_0 or weight_count == 1248424)
        assert (model != squeezeresnet_v1_1 or weight_count == 1235496)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
