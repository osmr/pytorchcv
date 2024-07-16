import torch
import torch.nn as nn
from typing import Callable
from common import (lambda_relu, lambda_batchnorm2d, lambda_instancenorm2d, lambda_groupnorm, conv1x1_block,
                    conv3x3_block, conv7x7_block)


class ResBlock(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit.

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
    normalization : function or None, default lambda_batchnorm2d()
        Lambda-function generator for normalization layer.
    final_activation : function or None, default None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | None = lambda_batchnorm2d(),
                 final_activation: Callable[..., nn.Module | None] | None = None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            normalization=normalization)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=bias,
            normalization=normalization,
            activation=final_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBottleneck(nn.Module):
    """
    ResNet bottleneck block for residual path in ResNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for the second convolution layer.
    dilation : int or tuple(int, int), default 1
        Dilation value for the second convolution layer.
    normalization : function or None, default lambda_batchnorm2d()
        Lambda-function generator for normalization layer.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    final_activation : function or None, default None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1,
                 normalization: Callable[..., nn.Module | None] | None = lambda_batchnorm2d(),
                 conv1_stride: bool = False,
                 bottleneck_factor: int = 4,
                 final_activation: Callable[..., nn.Module | None] | None = None):
        super(ResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1),
            normalization=normalization)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride),
            padding=padding,
            dilation=dilation,
            normalization=normalization)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            normalization=normalization,
            activation=final_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResUnit(nn.Module):
    """
    ResNet unit with residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple(int, int), default 1
        Dilation value for the second convolution layer in bottleneck.
    bias : bool, default False
        Whether the layer uses a bias vector.
    normalization : function or None, default lambda_batchnorm2d()
        Lambda-function generator for normalization layer.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    final_activation : function or None, default None
        Lambda-function generator for activation layer in the final convolution block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 padding: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = False,
                 normalization: Callable[..., nn.Module | None] | None = lambda_batchnorm2d(),
                 bottleneck: bool = True,
                 conv1_stride: bool = False,
                 final_activation: Callable[..., nn.Module | None] | None = None):
        super(ResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                normalization=normalization,
                conv1_stride=conv1_stride,
                final_activation=final_activation)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                normalization=normalization,
                final_activation=final_activation)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                normalization=normalization,
                activation=None)
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


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 planes,
                 norm_fn="group",
                 stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 planes,
                 norm_fn="group",
                 stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self,
                 output_dim=128,
                 norm_fn="batch",
                 dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        in_channels = 3
        init_block_channels = 64
        channels = [[64, 64], [96, 96], [128, 128]]
        bottleneck = False
        conv1_stride = True
        final_activation = lambda_relu()

        if self.norm_fn == "group":
            normalization = lambda_groupnorm(num_groups=8)
        elif self.norm_fn == "batch":
            normalization = lambda_batchnorm2d()
        elif self.norm_fn == "instance":
            normalization = lambda_instancenorm2d()
        elif self.norm_fn == "none":
            normalization = None
        else:
            assert False

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv7x7_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bias=True,
            normalization=normalization))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bias=True,
                    normalization=normalization,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride,
                    final_activation=final_activation))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)

        # self.in_planes = 64
        # self.layer1 = self._make_layer(64, stride=1)
        # self.layer2 = self._make_layer(96, stride=2)
        # self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.relu1(x)
        # x = self.init_block(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        x = self.features(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self,
                 output_dim=128,
                 norm_fn="batch",
                 dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        in_channels = 3
        init_block_channels = 32

        if self.norm_fn == "group":
            normalization = lambda_groupnorm(num_groups=8)
        elif self.norm_fn == "batch":
            normalization = lambda_batchnorm2d()
        elif self.norm_fn == "instance":
            normalization = lambda_instancenorm2d()
        elif self.norm_fn == "none":
            normalization = None
        else:
            assert False

        self.init_block = conv7x7_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bias=True,
            normalization=normalization)

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        # self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.relu1(x)
        x = self.init_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
