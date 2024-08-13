"""
    RiR for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Resnet in Resnet: Generalizing Residual Architectures,' https://arxiv.org/abs/1603.08029.
"""

__all__ = ['CIFARRiR', 'rir_cifar10', 'rir_cifar100', 'rir_svhn', 'RiRFinalBlock']

import os
import torch
import torch.nn as nn
from .common.conv import conv1x1, conv3x3, conv1x1_block, conv3x3_block
from .common.arch import DualPathSequential


class PostActivation(nn.Module):
    """
    Pure pre-activation block without convolution layer.

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


class RiRUnit(nn.Module):
    """
    RiR unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int]):
        super(RiRUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.res_pass_conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.trans_pass_conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.res_cross_conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.trans_cross_conv = conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.res_postactiv = PostActivation(in_channels=out_channels)
        self.trans_postactiv = PostActivation(in_channels=out_channels)
        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x_res, x_trans):
        if self.resize_identity:
            x_res_identity = self.identity_conv(x_res)
        else:
            x_res_identity = x_res

        y_res = self.res_cross_conv(x_res)
        y_trans = self.trans_cross_conv(x_trans)
        x_res = self.res_pass_conv(x_res)
        x_trans = self.trans_pass_conv(x_trans)

        x_res = x_res + x_res_identity + y_trans
        x_trans = x_trans + y_res

        x_res = self.res_postactiv(x_res)
        x_trans = self.trans_postactiv(x_trans)

        return x_res, x_trans


class RiRInitBlock(nn.Module):
    """
    RiR initial block.

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
        super(RiRInitBlock, self).__init__()
        self.res_conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.trans_conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x, _):
        x_res = self.res_conv(x)
        x_trans = self.trans_conv(x)
        return x_res, x_trans


class RiRFinalBlock(nn.Module):
    """
    RiR final block.
    """
    def __init__(self):
        super(RiRFinalBlock, self).__init__()

    def forward(self, x_res, x_trans):
        x = torch.cat((x_res, x_trans), dim=1)
        return x, None


class CIFARRiR(nn.Module):
    """
    RiR model for CIFAR from 'Resnet in Resnet: Generalizing Residual Architectures,' https://arxiv.org/abs/1603.08029.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (32, 32)
        Spatial size of the expected input image.
    num_classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels: list[list[int]],
                 init_block_channels: int,
                 final_block_channels: int,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (32, 32),
                 num_classes: int = 10):
        super(CIFARRiR, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=0,
            last_ordinals=0)
        self.features.add_module("init_block", RiRInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), RiRUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", RiRFinalBlock())
        in_channels = final_block_channels

        self.output = nn.Sequential()
        self.output.add_module("final_conv", conv1x1_block(
            in_channels=in_channels,
            out_channels=num_classes,
            activation=None))
        self.output.add_module("final_pool", nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_rir_cifar(num_classes: int,
                  model_name: str | None = None,
                  pretrained: bool = False,
                  root: str = os.path.join("~", ".torch", "models"),
                  **kwargs) -> nn.Module:
    """
    Create RiR model for CIFAR with specific parameters.

    Parameters
    ----------
    num_classes : int
        Number of classification classes.
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
    channels = [[48, 48, 48, 48], [96, 96, 96, 96, 96, 96], [192, 192, 192, 192, 192, 192]]
    init_block_channels = 48
    final_block_channels = 384

    net = CIFARRiR(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        num_classes=num_classes,
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


def rir_cifar10(num_classes: int = 10,
                **kwargs) -> nn.Module:
    """
    RiR model for CIFAR-10 from 'Resnet in Resnet: Generalizing Residual Architectures,'
    https://arxiv.org/abs/1603.08029.

    Parameters
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_rir_cifar(
        num_classes=num_classes,
        model_name="rir_cifar10",
        **kwargs)


def rir_cifar100(num_classes: int = 100,
                 **kwargs) -> nn.Module:
    """
    RiR model for CIFAR-100 from 'Resnet in Resnet: Generalizing Residual Architectures,'
    https://arxiv.org/abs/1603.08029.

    Parameters
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_rir_cifar(
        num_classes=num_classes,
        model_name="rir_cifar100",
        **kwargs)


def rir_svhn(num_classes: int = 10,
             **kwargs) -> nn.Module:
    """
    RiR model for SVHN from 'Resnet in Resnet: Generalizing Residual Architectures,'
    https://arxiv.org/abs/1603.08029.

    Parameters
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_rir_cifar(
        num_classes=num_classes,
        model_name="rir_svhn",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        (rir_cifar10, 10),
        (rir_cifar100, 100),
        (rir_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != rir_cifar10 or weight_count == 9492980)
        assert (model != rir_cifar100 or weight_count == 9527720)
        assert (model != rir_svhn or weight_count == 9492980)

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
