"""
    IBPPose for COCO Keypoint, implemented in PyTorch.
    Original paper: 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation,'
    https://arxiv.org/abs/1911.10529.
"""

__all__ = ['IbpPose', 'ibppose_coco']

import os
import torch
from torch import nn
from typing import Callable
from .common.activ import lambda_relu, lambda_leakyrelu, create_activation_layer
from .common.norm import lambda_batchnorm2d
from .common.conv import conv1x1_block, conv3x3_block, conv7x7_block
from .common.arch import Hourglass
from .common.att import SEBlock
from .common.tutti import InterpolationBlock


class IbpResBottleneck(nn.Module):
    """
    Bottleneck block for residual path in the residual unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function, default lambda_relu()
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 bias: bool = False,
                 bottleneck_factor: int = 2,
                 activation: Callable[..., nn.Module] = lambda_relu()):
        super(IbpResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias,
            activation=activation)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            bias=bias,
            activation=activation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IbpResUnit(nn.Module):
    """
    ResNet-like residual unit with residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int), default 1
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function, default lambda_relu()
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int] = 1,
                 bias: bool = False,
                 bottleneck_factor: int = 2,
                 activation: Callable[..., nn.Module] = lambda_relu()):
        super(IbpResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = IbpResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            bottleneck_factor=bottleneck_factor,
            activation=activation)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                activation=None)
        self.activ = create_activation_layer(activation)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class IbpBackbone(nn.Module):
    """
    IBPPose backbone.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable[..., nn.Module]):
        super(IbpBackbone, self).__init__()
        dilations = (3, 3, 4, 4, 5, 5)
        mid1_channels = out_channels // 4
        mid2_channels = out_channels // 2

        self.conv1 = conv7x7_block(
            in_channels=in_channels,
            out_channels=mid1_channels,
            stride=2,
            activation=activation)
        self.res1 = IbpResUnit(
            in_channels=mid1_channels,
            out_channels=mid2_channels,
            activation=activation)
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.res2 = IbpResUnit(
            in_channels=mid2_channels,
            out_channels=mid2_channels,
            activation=activation)
        self.dilation_branch = nn.Sequential()
        for i, dilation in enumerate(dilations):
            self.dilation_branch.add_module("block{}".format(i + 1), conv3x3_block(
                in_channels=mid2_channels,
                out_channels=mid2_channels,
                padding=dilation,
                dilation=dilation,
                activation=activation))

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        y = self.dilation_branch(x)
        x = torch.cat((x, y), dim=1)
        return x


class IbpDownBlock(nn.Module):
    """
    IBPPose down block for the hourglass.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable[..., nn.Module]):
        super(IbpDownBlock, self).__init__()
        self.down = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.res = IbpResUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation)

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class IbpUpBlock(nn.Module):
    """
    IBPPose up block for the hourglass.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether some layers use a bias vector.
    normalization : function or None
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool,
                 normalization: Callable[..., nn.Module] | None,
                 activation: Callable[..., nn.Module]):
        super(IbpUpBlock, self).__init__()
        self.res = IbpResUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation)
        self.up = InterpolationBlock(
            scale_factor=2,
            mode="nearest",
            align_corners=None)
        self.conv = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=bias,
            normalization=normalization,
            activation=activation)

    def forward(self, x):
        x = self.res(x)
        x = self.up(x)
        x = self.conv(x)
        return x


class MergeBlock(nn.Module):
    """
    IBPPose merge block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether some layers use a bias vector.
    normalization : function or None
        Lambda-function generator for normalization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool,
                 normalization: Callable[..., nn.Module] | None):
        super(MergeBlock, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            normalization=normalization,
            activation=None)

    def forward(self, x):
        return self.conv(x)


class IbpPreBlock(nn.Module):
    """
    IBPPose preliminary decoder block.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    bias : bool
        Whether some layers use a bias vector.
    normalization : function or None
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 out_channels: int,
                 bias: bool,
                 normalization: Callable[..., nn.Module] | None,
                 activation: Callable[..., nn.Module]):
        super(IbpPreBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=bias,
            normalization=normalization,
            activation=activation)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=bias,
            normalization=normalization,
            activation=activation)
        self.se = SEBlock(
            channels=out_channels,
            use_conv=False,
            mid_activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x


class IbpPass(nn.Module):
    """
    IBPPose single pass decoder block.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    mid_channels : int
        Number of middle channels.
    depth : int
        Depth of hourglass.
    growth_rate : int
        Addition for number of channel for each level.
    bias : bool
        Whether some layers use a bias vector.
    normalization : function or None
        Lambda-function generator for normalization layer.
    activation : function
        Lambda-function generator for activation layer.
    """
    def __init__(self,
                 channels: int,
                 mid_channels: int,
                 depth: int,
                 growth_rate: int,
                 merge: int,
                 bias: bool,
                 normalization: Callable[..., nn.Module] | None,
                 activation: Callable[..., nn.Module]):
        super(IbpPass, self).__init__()
        self.merge = merge

        down_seq = nn.Sequential()
        up_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        top_channels = channels
        bottom_channels = channels
        for i in range(depth + 1):
            skip_seq.add_module("skip{}".format(i + 1), IbpResUnit(
                in_channels=top_channels,
                out_channels=top_channels,
                activation=activation))
            bottom_channels += growth_rate
            if i < depth:
                down_seq.add_module("down{}".format(i + 1), IbpDownBlock(
                    in_channels=top_channels,
                    out_channels=bottom_channels,
                    activation=activation))
                up_seq.add_module("up{}".format(i + 1), IbpUpBlock(
                    in_channels=bottom_channels,
                    out_channels=top_channels,
                    bias=bias,
                    normalization=normalization,
                    activation=activation))
            top_channels = bottom_channels
        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            return_first_skip=False)

        self.pre_block = IbpPreBlock(
            out_channels=channels,
            bias=bias,
            normalization=normalization,
            activation=activation)
        self.post_block = conv1x1_block(
            in_channels=channels,
            out_channels=mid_channels,
            bias=True,
            normalization=None,
            activation=None)

        if self.merge:
            self.pre_merge_block = MergeBlock(
                in_channels=channels,
                out_channels=channels,
                bias=bias,
                normalization=normalization)
            self.post_merge_block = MergeBlock(
                in_channels=mid_channels,
                out_channels=channels,
                bias=bias,
                normalization=normalization)

    def forward(self, x, x_prev):
        x = self.hg(x)
        if x_prev is not None:
            x = x + x_prev
        y = self.pre_block(x)
        z = self.post_block(y)
        if self.merge:
            z = self.post_merge_block(z) + self.pre_merge_block(y)
        return z


class IbpPose(nn.Module):
    """
    IBPPose model from 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation,'
    https://arxiv.org/abs/1911.10529.

    Parameters
    ----------
    passes : int
        Number of passes.
    backbone_out_channels : int
        Number of output channels for the backbone.
    outs_channels : int
        Number of output channels for the backbone.
    depth : int
        Depth of hourglass.
    growth_rate : int
        Addition for number of channel for each level.
    use_bn : bool
        Whether to use BatchNorm layer.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (256, 256)
        Spatial size of the expected input image.
    """
    def __init__(self,
                 passes: int,
                 backbone_out_channels: int,
                 outs_channels: int,
                 depth: int,
                 growth_rate: int,
                 use_bn: bool,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (256, 256)):
        super(IbpPose, self).__init__()
        self.in_size = in_size
        bias = (not use_bn)
        normalization = lambda_batchnorm2d() if use_bn else None
        activation = lambda_leakyrelu()

        self.backbone = IbpBackbone(
            in_channels=in_channels,
            out_channels=backbone_out_channels,
            activation=activation)

        self.decoder = nn.Sequential()
        for i in range(passes):
            merge = (i != passes - 1)
            self.decoder.add_module("pass{}".format(i + 1), IbpPass(
                channels=backbone_out_channels,
                mid_channels=outs_channels,
                depth=depth,
                growth_rate=growth_rate,
                merge=merge,
                bias=bias,
                normalization=normalization,
                activation=activation))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.backbone(x)
        x_prev = None
        for module in self.decoder._modules.values():
            if x_prev is not None:
                x = x + x_prev
            x_prev = module(x, x_prev)
        return x_prev


def get_ibppose(model_name: str | None = None,
                pretrained: bool = False,
                root: str = os.path.join("~", ".torch", "models"),
                **kwargs) -> nn.Module:
    """
    Create IBPPose model with specific parameters.

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
    passes = 4
    backbone_out_channels = 256
    outs_channels = 50
    depth = 4
    growth_rate = 128
    use_bn = True

    net = IbpPose(
        passes=passes,
        backbone_out_channels=backbone_out_channels,
        outs_channels=outs_channels,
        depth=depth,
        growth_rate=growth_rate,
        use_bn=use_bn,
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


def ibppose_coco(**kwargs) -> nn.Module:
    """
    IBPPose model for COCO Keypoint from 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person
    Pose Estimation,' https://arxiv.org/abs/1911.10529.

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
    return get_ibppose(
        model_name="ibppose_coco",
        **kwargs)


def _test():
    from .common.model_store import calc_net_weight_count

    in_size = (256, 256)
    pretrained = False

    models = [
        ibppose_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibppose_coco or weight_count == 95827784)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        assert ((y.shape[0] == batch) and (y.shape[1] == 50))
        assert ((y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4))


if __name__ == "__main__":
    _test()
