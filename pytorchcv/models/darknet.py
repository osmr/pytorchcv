"""
    DarkNet for ImageNet-1K, implemented in PyTorch.
    Original source: 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.
"""

__all__ = ['DarkNet', 'darknet_ref', 'darknet_tiny', 'darknet19']

import os
import torch
import torch.nn as nn
from typing import Callable
from .common.activ import lambda_leakyrelu, create_activation_layer
from .common.conv import conv1x1_block, conv3x3_block


def dark_convYxY(in_channels: int,
                 out_channels: int,
                 activation: Callable[..., nn.Module],
                 pointwise: bool):
    """
    DarkNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function
        Lambda-function generator for activation layer.
    pointwise : bool
        Whether to use 1x1 (pointwise) convolution or 3x3 convolution.
    """
    if pointwise:
        return conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation)
    else:
        return conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation)


class DarkNet(nn.Module):
    """
    DarkNet model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    odd_pointwise : bool
        Whether pointwise convolution layer is used for each odd unit.
    avg_pool_size : int
        Window size of the final average pooling.
    cls_activ : bool
        Whether classification convolution layer uses an activation.
    alpha : float, default 0.1
        Slope coefficient for Leaky ReLU activation.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels: list[list[int]],
                 odd_pointwise: bool,
                 avg_pool_size: int,
                 cls_activ: bool,
                 alpha: float = 0.1,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (224, 224),
                 num_classes: int = 1000):
        super(DarkNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        activation = lambda_leakyrelu(negative_slope=alpha)

        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), dark_convYxY(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    activation=activation,
                    pointwise=(len(channels_per_stage) > 1) and not (((j + 1) % 2 == 1) ^ odd_pointwise)))
                in_channels = out_channels
            if i != len(channels) - 1:
                stage.add_module("pool{}".format(i + 1), nn.MaxPool2d(
                    kernel_size=2,
                    stride=2))
            self.features.add_module("stage{}".format(i + 1), stage)

        self.output = nn.Sequential()
        self.output.add_module("final_conv", nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1))
        if cls_activ:
            self.output.add_module("final_activ", create_activation_layer(activation))
        self.output.add_module("final_pool", nn.AvgPool2d(
            kernel_size=avg_pool_size,
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


def get_darknet(version: str,
                model_name: str | None = None,
                pretrained: bool = False,
                root: str = os.path.join("~", ".torch", "models"),
                **kwargs) -> nn.Module:
    """
    Create DarkNet model with specific parameters.

    Parameters
    ----------
    version : str
        Version of SqueezeNet ('ref', 'tiny' or '19').
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
    if version == "ref":
        channels = [[16], [32], [64], [128], [256], [512], [1024]]
        odd_pointwise = False
        avg_pool_size = 3
        cls_activ = True
    elif version == "tiny":
        channels = [[16], [32], [16, 128, 16, 128], [32, 256, 32, 256], [64, 512, 64, 512, 128]]
        odd_pointwise = True
        avg_pool_size = 14
        cls_activ = False
    elif version == "19":
        channels = [[32], [64], [128, 64, 128], [256, 128, 256], [512, 256, 512, 256, 512],
                    [1024, 512, 1024, 512, 1024]]
        odd_pointwise = False
        avg_pool_size = 7
        cls_activ = False
    else:
        raise ValueError("Unsupported DarkNet version {}".format(version))

    net = DarkNet(
        channels=channels,
        odd_pointwise=odd_pointwise,
        avg_pool_size=avg_pool_size,
        cls_activ=cls_activ,
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


def darknet_ref(**kwargs) -> nn.Module:
    """
    DarkNet 'Reference' model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

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
    return get_darknet(
        version="ref",
        model_name="darknet_ref",
        **kwargs)


def darknet_tiny(**kwargs) -> nn.Module:
    """
    DarkNet Tiny model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

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
    return get_darknet(
        version="tiny",
        model_name="darknet_tiny",
        **kwargs)


def darknet19(**kwargs) -> nn.Module:
    """
    DarkNet-19 model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

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
    return get_darknet(
        version="19",
        model_name="darknet19",
        **kwargs)


def _test():
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        darknet_ref,
        darknet_tiny,
        darknet19,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != darknet_ref or weight_count == 7319416)
        assert (model != darknet_tiny or weight_count == 1042104)
        assert (model != darknet19 or weight_count == 20842376)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
