"""
    WRN-1bit for CIFAR/SVHN, implemented in PyTorch.
    Original paper: 'Training wide residual networks for deployment using a single bit for each weight,'
    https://arxiv.org/abs/1802.08530.
"""

__all__ = ['CIFARWRN1bit', 'wrn20_10_1bit_cifar10', 'wrn20_10_1bit_cifar100', 'wrn20_10_1bit_svhn',
           'wrn20_10_32bit_cifar10', 'wrn20_10_32bit_cifar100', 'wrn20_10_32bit_svhn']

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Binarize(torch.autograd.Function):
    """
    Fake sign op for 1-bit weights.
    """

    @staticmethod
    def forward(ctx, x):
        return math.sqrt(2.0 / (x.shape[1] * x.shape[2] * x.shape[3])) * x.sign()

    @staticmethod
    def backward(ctx, dy):
        return dy


class Conv2d1bit(nn.Conv2d):
    """
    Standard convolution block with binarization.

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
    padding : int or tuple(int, int), default 1
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 binarized: bool = False):
        super(Conv2d1bit, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.binarized = binarized

    def forward(self, input):
        weight = Binarize.apply(self.weight) if self.binarized else self.weight
        bias = Binarize.apply(self.bias) if self.bias is not None and self.binarized else self.bias
        return F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)


def conv1x1_1bit(in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 binarized: bool = False):
    """
    Convolution 1x1 layer with binarization.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    binarized : bool, default False
        Whether to use binarization.
    """
    return Conv2d1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias,
        binarized=binarized)


def conv3x3_1bit(in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 binarized: bool = False):
    """
    Convolution 3x3 layer with binarization.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    binarized : bool, default False
        Whether to use binarization.
    """
    return Conv2d1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        binarized=binarized)


class ConvBlock1bit(nn.Module):
    """
    Standard convolution block with Batch normalization and ReLU activation, and binarization.

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
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 bn_affine: bool = True,
                 activate: bool = True,
                 binarized: bool = False):
        super(ConvBlock1bit, self).__init__()
        self.activate = activate

        self.conv = Conv2d1bit(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            binarized=binarized)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            affine=bn_affine)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block_1bit(in_channels: int,
                       out_channels: int,
                       stride: int | tuple[int, int] = 1,
                       padding: int | tuple[int, int] = 0,
                       groups: int = 1,
                       bias: bool = False,
                       bn_affine: bool = True,
                       activate: bool = True,
                       binarized: bool = False):
    """
    1x1 version of the standard convolution block with binarization.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int), default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    return ConvBlock1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        bn_affine=bn_affine,
        activate=activate,
        binarized=binarized)


class PreConvBlock1bit(nn.Module):
    """
    Convolution block with Batch normalization and ReLU pre-activation, and binarization.

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
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int],
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = False,
                 bn_affine: bool = True,
                 return_preact: bool = False,
                 activate: bool = True,
                 binarized: bool = False):
        super(PreConvBlock1bit, self).__init__()
        self.return_preact = return_preact
        self.activate = activate

        self.bn = nn.BatchNorm2d(
            num_features=in_channels,
            affine=bn_affine)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)
        self.conv = Conv2d1bit(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            binarized=binarized)

    def forward(self, x):
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_conv3x3_block_1bit(in_channels: int,
                           out_channels: int,
                           stride: int | tuple[int, int] = 1,
                           padding: int | tuple[int, int] = 1,
                           dilation: int | tuple[int, int] = 1,
                           bn_affine: bool = True,
                           return_preact: bool = False,
                           activate: bool = True,
                           binarized: bool = False):
    """
    3x3 version of the pre-activated convolution block with binarization.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for convolution layer.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    return PreConvBlock1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bn_affine=bn_affine,
        return_preact=return_preact,
        activate=activate,
        binarized=binarized)


class PreResBlock1bit(nn.Module):
    """
    Simple PreResNet block for residual path in ResNet unit (with binarization).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 binarized: bool = False):
        super(PreResBlock1bit, self).__init__()
        self.conv1 = pre_conv3x3_block_1bit(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bn_affine=False,
            return_preact=False,
            binarized=binarized)
        self.conv2 = pre_conv3x3_block_1bit(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_affine=False,
            binarized=binarized)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PreResUnit1bit(nn.Module):
    """
    PreResNet unit with residual connection (with binarization).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 binarized: bool = False):
        super(PreResUnit1bit, self).__init__()
        self.resize_identity = (stride != 1)

        self.body = PreResBlock1bit(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            binarized=binarized)
        if self.resize_identity:
            self.identity_pool = nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1)

    def forward(self, x):
        identity = x
        x = self.body(x)
        if self.resize_identity:
            identity = self.identity_pool(identity)
            identity = torch.cat((identity, torch.zeros_like(identity)), dim=1)
        x = x + identity
        return x


class PreResActivation(nn.Module):
    """
    PreResNet pure pre-activation block without convolution layer. It's used by itself as the final block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    """
    def __init__(self,
                 in_channels: int,
                 bn_affine: bool = True):
        super(PreResActivation, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features=in_channels,
            affine=bn_affine)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class CIFARWRN1bit(nn.Module):
    """
    WRN-1bit model for CIFAR from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    binarized : bool, default True
        Whether to use binarization.
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
                 binarized: bool = True,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (32, 32),
                 num_classes: int = 10):
        super(CIFARWRN1bit, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_1bit(
            in_channels=in_channels,
            out_channels=init_block_channels,
            binarized=binarized))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), PreResUnit1bit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    binarized=binarized))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("post_activ", PreResActivation(
            in_channels=in_channels,
            bn_affine=False))

        self.output = nn.Sequential()
        self.output.add_module("final_conv", conv1x1_block_1bit(
            in_channels=in_channels,
            out_channels=num_classes,
            activate=False,
            binarized=binarized))
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


def get_wrn1bit_cifar(num_classes: int,
                      blocks: int,
                      width_factor: int,
                      binarized: bool = True,
                      model_name: str | None = None,
                      pretrained: bool = False,
                      root: str = os.path.join("~", ".torch", "models"),
                      **kwargs) -> nn.Module:
    """
    Create WRN-1bit model for CIFAR with specific parameters.

    Parameters
    ----------
    num_classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    width_factor : int
        Wide scale factor for width of layers.
    binarized : bool, default True
        Whether to use binarization.
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
    assert ((blocks - 2) % 6 == 0)
    layers = [(blocks - 2) // 6] * 3
    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci * width_factor] * li for (ci, li) in zip(channels_per_layers, layers)]
    init_block_channels *= width_factor

    net = CIFARWRN1bit(
        channels=channels,
        init_block_channels=init_block_channels,
        binarized=binarized,
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


def wrn20_10_1bit_cifar10(num_classes: int = 10,
                          **kwargs) -> nn.Module:
    """
    WRN-20-10-1bit model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

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
    return get_wrn1bit_cifar(
        num_classes=num_classes,
        blocks=20,
        width_factor=10,
        binarized=True,
        model_name="wrn20_10_1bit_cifar10",
        **kwargs)


def wrn20_10_1bit_cifar100(num_classes: int = 100,
                           **kwargs) -> nn.Module:
    """
    WRN-20-10-1bit model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

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
    return get_wrn1bit_cifar(
        num_classes=num_classes,
        blocks=20,
        width_factor=10,
        binarized=True,
        model_name="wrn20_10_1bit_cifar100",
        **kwargs)


def wrn20_10_1bit_svhn(num_classes: int = 10,
                       **kwargs) -> nn.Module:
    """
    WRN-20-10-1bit model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

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
    return get_wrn1bit_cifar(
        num_classes=num_classes,
        blocks=20,
        width_factor=10,
        binarized=True,
        model_name="wrn20_10_1bit_svhn",
        **kwargs)


def wrn20_10_32bit_cifar10(num_classes: int = 10,
                           **kwargs) -> nn.Module:
    """
    WRN-20-10-32bit model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

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
    return get_wrn1bit_cifar(
        num_classes=num_classes,
        blocks=20,
        width_factor=10,
        binarized=False,
        model_name="wrn20_10_32bit_cifar10",
        **kwargs)


def wrn20_10_32bit_cifar100(num_classes: int = 100,
                            **kwargs) -> nn.Module:
    """
    WRN-20-10-32bit model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

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
    return get_wrn1bit_cifar(
        num_classes=num_classes,
        blocks=20,
        width_factor=10,
        binarized=False,
        model_name="wrn20_10_32bit_cifar100",
        **kwargs)


def wrn20_10_32bit_svhn(num_classes: int = 10,
                        **kwargs) -> nn.Module:
    """
    WRN-20-10-32bit model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

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
    return get_wrn1bit_cifar(
        num_classes=num_classes,
        blocks=20,
        width_factor=10,
        binarized=False,
        model_name="wrn20_10_32bit_svhn",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        (wrn20_10_1bit_cifar10, 10),
        (wrn20_10_1bit_cifar100, 100),
        (wrn20_10_1bit_svhn, 10),
        (wrn20_10_32bit_cifar10, 10),
        (wrn20_10_32bit_cifar100, 100),
        (wrn20_10_32bit_svhn, 10),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn20_10_1bit_cifar10 or weight_count == 26737140)
        assert (model != wrn20_10_1bit_cifar100 or weight_count == 26794920)
        assert (model != wrn20_10_1bit_svhn or weight_count == 26737140)
        assert (model != wrn20_10_32bit_cifar10 or weight_count == 26737140)
        assert (model != wrn20_10_32bit_cifar100 or weight_count == 26794920)
        assert (model != wrn20_10_32bit_svhn or weight_count == 26737140)

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
