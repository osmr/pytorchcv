"""
    AlexNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.
"""

__all__ = ['AlexNet', 'alexnet', 'alexnetb']

import os
import torch.nn as nn
import torch.nn.functional as F
from .common.conv import ConvBlock


class AlexConv(ConvBlock):
    """
    AlexNet specific convolution block.

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
    use_lrn : bool
        Whether to use LRN layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 use_lrn: bool):
        super(AlexConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            normalization=None)
        self.use_lrn = use_lrn

    def forward(self, x):
        x = super(AlexConv, self).forward(x)
        if self.use_lrn:
            x = F.local_response_norm(x, size=5, k=2.0)
        return x


class AlexDense(nn.Module):
    """
    AlexNet specific dense block.

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
        super(AlexDense, self).__init__()
        self.fc = nn.Linear(
            in_features=in_channels,
            out_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.activ(x)
        x = self.dropout(x)
        return x


class AlexOutputBlock(nn.Module):
    """
    AlexNet specific output block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels: int,
                 classes: int):
        super(AlexOutputBlock, self).__init__()
        mid_channels = 4096

        self.fc1 = AlexDense(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.fc2 = AlexDense(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.fc3 = nn.Linear(
            in_features=mid_channels,
            out_features=classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    kernel_sizes : list(list(int))
        Convolution window sizes for each unit.
    strides : list(list(int))
        Strides of the convolution for each unit.
    paddings : list(list(int))
        Padding value for convolution layer for each unit.
    use_lrn : bool
        Whether to use LRN layer.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels: list[list[int]],
                 kernel_sizes: list[list[int]],
                 strides: list[list[int]],
                 paddings: list[list[int]],
                 use_lrn: bool,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (224, 224),
                 num_classes: int = 1000):
        super(AlexNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            use_lrn_i = use_lrn and (i in [0, 1])
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), AlexConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i][j],
                    stride=strides[i][j],
                    padding=paddings[i][j],
                    use_lrn=use_lrn_i))
                in_channels = out_channels
            stage.add_module("pool{}".format(i + 1), nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=0,
                ceil_mode=True))
            self.features.add_module("stage{}".format(i + 1), stage)

        self.output = AlexOutputBlock(
            in_channels=(in_channels * 6 * 6),
            classes=num_classes)

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


def get_alexnet(version: str = "a",
                model_name: str | None = None,
                pretrained: bool = False,
                root: str = os.path.join("~", ".torch", "models"),
                **kwargs) -> nn.Module:
    """
    Create AlexNet model with specific parameters.

    Parameters
    ----------
    version : str, default 'a'
        Version of AlexNet ('a' or 'b').
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
    if version == "a":
        channels = [[96], [256], [384, 384, 256]]
        kernel_sizes = [[11], [5], [3, 3, 3]]
        strides = [[4], [1], [1, 1, 1]]
        paddings = [[0], [2], [1, 1, 1]]
        use_lrn = True
    elif version == "b":
        channels = [[64], [192], [384, 256, 256]]
        kernel_sizes = [[11], [5], [3, 3, 3]]
        strides = [[4], [1], [1, 1, 1]]
        paddings = [[2], [2], [1, 1, 1]]
        use_lrn = False
    else:
        raise ValueError("Unsupported AlexNet version {}".format(version))

    net = AlexNet(
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        use_lrn=use_lrn,
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


def alexnet(**kwargs) -> nn.Module:
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

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
    return get_alexnet(
        model_name="alexnet",
        **kwargs)


def alexnetb(**kwargs) -> nn.Module:
    """
    AlexNet-b model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997. Non-standard version.

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
    return get_alexnet(
        version="b",
        model_name="alexnetb",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count, get_model_weight_count

    pretrained = False

    models = [
        alexnet,
        alexnetb,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (weight_count == get_model_weight_count(model.__name__))

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
