"""
    ICNet for image segmentation, implemented in PyTorch.
    Original paper: 'ICNet for Real-Time Semantic Segmentation on High-Resolution Images,'
    https://arxiv.org/abs/1704.08545.
"""

__all__ = ['ICNet', 'icnet_resnetd50b_cityscapes']

import os
import torch.nn as nn
from .common.conv import conv1x1, conv1x1_block, conv3x3_block
from .common.arch import MultiOutputSequential
from .common.tutti import InterpolationBlock
from .pspnet import PyramidPooling
from .resnetd import resnetd50b


class ICInitBlock(nn.Module):
    """
    ICNet specific initial block.

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
        super(ICInitBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=2)
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PSPBlock(nn.Module):
    """
    ICNet specific PSPNet reduced head block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    upscale_out_size : tuple(int, int)
        Spatial size of the input tensor for the bilinear upsampling operation.
    bottleneck_factor : int
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels: int,
                 upscale_out_size: tuple[int, int],
                 bottleneck_factor: int):
        super(PSPBlock, self).__init__()
        assert (in_channels % bottleneck_factor == 0)
        mid_channels = in_channels // bottleneck_factor

        self.pool = PyramidPooling(
            in_channels=in_channels,
            upscale_out_size=upscale_out_size)
        self.conv = conv3x3_block(
            in_channels=4096,
            out_channels=mid_channels)
        self.dropout = nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class CFFBlock(nn.Module):
    """
    Cascade Feature Fusion block.

    Parameters
    ----------
    in_channels_low : int
        Number of input channels (low input).
    in_channels_high : int
        Number of input channels (low high).
    out_channels : int
        Number of output channels.
    num_classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels_low: int,
                 in_channels_high: int,
                 out_channels: int,
                 num_classes: int):
        super(CFFBlock, self).__init__()
        self.up = InterpolationBlock(scale_factor=2)
        self.conv_low = conv3x3_block(
            in_channels=in_channels_low,
            out_channels=out_channels,
            padding=2,
            dilation=2,
            activation=None)
        self.conv_hign = conv1x1_block(
            in_channels=in_channels_high,
            out_channels=out_channels,
            activation=None)
        self.activ = nn.ReLU(inplace=True)
        self.conv_cls = conv1x1(
            in_channels=out_channels,
            out_channels=num_classes)

    def forward(self, xl, xh):
        xl = self.up(xl)
        xl = self.conv_low(xl)
        xh = self.conv_hign(xh)
        x = xl + xh
        x = self.activ(x)
        x_cls = self.conv_cls(xl)
        return x, x_cls


class ICHeadBlock(nn.Module):
    """
    ICNet head block.

    Parameters
    ----------
    num_classes : int
        Number of classification classes.
    """
    def __init__(self,
                 num_classes: int):
        super(ICHeadBlock, self).__init__()
        self.cff_12 = CFFBlock(
            in_channels_low=128,
            in_channels_high=64,
            out_channels=128,
            num_classes=num_classes)
        self.cff_24 = CFFBlock(
            in_channels_low=256,
            in_channels_high=256,
            out_channels=128,
            num_classes=num_classes)
        self.up_x2 = InterpolationBlock(scale_factor=2)
        self.up_x8 = InterpolationBlock(scale_factor=4)
        self.conv_cls = conv1x1(
            in_channels=128,
            out_channels=num_classes)

    def forward(self, x1, x2, x4):
        outputs = []

        x_cff_24, x_24_cls = self.cff_24(x4, x2)
        outputs.append(x_24_cls)

        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x1)
        outputs.append(x_12_cls)

        up_x2 = self.up_x2(x_cff_12)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)

        up_x8 = self.up_x8(up_x2)
        outputs.append(up_x8)

        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return tuple(outputs)


class ICNet(nn.Module):
    """
    ICNet model from 'ICNet for Real-Time Semantic Segmentation on High-Resolution Images,'
    https://arxiv.org/abs/1704.08545.

    Parameters
    ----------
    backbones : tuple(nn.Sequential, nn.Sequential)
        Feature extractors.
    backbones_out_channels : tuple(int, int)
        Number of output channels form each feature extractor.
    channels : tuple(int, int, int)
        Number of output channels for each branch.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (480, 480)
        Spatial size of the expected input image.
    num_classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 backbones: tuple[nn.Sequential, nn.Sequential],
                 backbones_out_channels: tuple[int, int],
                 channels: tuple[int, int, int],
                 aux: bool = False,
                 fixed_size: bool = True,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (480, 480),
                 num_classes: int = 21):
        super(ICNet, self).__init__()
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.aux = aux
        self.fixed_size = fixed_size
        psp_pool_out_size = (self.in_size[0] // 32, self.in_size[1] // 32) if fixed_size else None
        psp_head_out_channels = 512

        self.branch1 = ICInitBlock(
            in_channels=in_channels,
            out_channels=channels[0])

        self.branch2 = MultiOutputSequential()
        self.branch2.add_module("down1", InterpolationBlock(scale_factor=2, up=False))
        backbones[0].do_output = True
        self.branch2.add_module("backbones1", backbones[0])

        self.branch2.add_module("down2", InterpolationBlock(scale_factor=2, up=False))
        self.branch2.add_module("backbones2", backbones[1])
        self.branch2.add_module("psp", PSPBlock(
            in_channels=backbones_out_channels[1],
            upscale_out_size=psp_pool_out_size,
            bottleneck_factor=4))
        self.branch2.add_module("final_block", conv1x1_block(
            in_channels=psp_head_out_channels,
            out_channels=channels[2]))

        self.conv_y2 = conv1x1_block(
            in_channels=backbones_out_channels[0],
            out_channels=channels[1])

        self.final_block = ICHeadBlock(num_classes=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        y1 = self.branch1(x)
        y3, y2 = self.branch2(x)
        y2 = self.conv_y2(y2)
        x = self.final_block(y1, y2, y3)
        if self.aux:
            return x
        else:
            return x[0]


def get_icnet(backbones: tuple[nn.Sequential, nn.Sequential],
              backbones_out_channels: tuple[int, int],
              num_classes: int,
              aux: bool = False,
              model_name: str | None = None,
              pretrained: bool = False,
              root: str = os.path.join("~", ".torch", "models"),
              **kwargs) -> nn.Module:
    """
    Create ICNet model with specific parameters.

    Parameters
    ----------
    backbones : tuple of nn.Sequential
        Feature extractors.
    backbones_out_channels : tuple of int
        Number of output channels form each feature extractor.
    num_classes : int
        Number of segmentation classes.
    aux : bool, default False
        Whether to output an auxiliary result.
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
    channels = (64, 256, 256)

    backbones[0].multi_output = False
    backbones[1].multi_output = False

    net = ICNet(
        backbones=backbones,
        backbones_out_channels=backbones_out_channels,
        channels=channels,
        num_classes=num_classes,
        aux=aux,
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


def icnet_resnetd50b_cityscapes(pretrained_backbone: bool = False,
                                num_classes: int = 19,
                                aux: bool = True,
                                **kwargs) -> nn.Module:
    """
    ICNet model on the base of ResNet(D)-50b for Cityscapes from 'ICNet for Real-Time Semantic Segmentation on
    High-Resolution Images,' https://arxiv.org/abs/1704.08545.

    Parameters
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    backbone1 = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=None).features
    for i in range(len(backbone1) - 3):
        del backbone1[-1]
    backbone2 = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=None).features
    del backbone2[-1]
    for i in range(3):
        del backbone2[0]
    backbones = (backbone1, backbone2)
    backbones_out_channels = (512, 2048)
    return get_icnet(
        backbones=backbones,
        backbones_out_channels=backbones_out_channels,
        num_classes=num_classes,
        aux=aux,
        model_name="icnet_resnetd50b_cityscapes",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    in_size = (480, 480)
    aux = False
    fixed_size = False
    pretrained = False

    models = [
        (icnet_resnetd50b_cityscapes, 19),
    ]

    for model, num_classes in models:

        net = model(
            pretrained=pretrained,
            in_size=in_size,
            fixed_size=fixed_size,
            aux=aux)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != icnet_resnetd50b_cityscapes or weight_count == 47489184)

        x = torch.randn(1, 3, in_size[0], in_size[1])
        ys = net(x)
        y = ys[0] if aux else ys
        y.sum().backward()
        assert ((y.size(0) == x.size(0)) and (y.size(1) == num_classes) and (y.size(2) == x.size(2)) and
                (y.size(3) == x.size(3)))


if __name__ == "__main__":
    _test()
