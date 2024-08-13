"""
    SegNet for image segmentation, implemented in PyTorch.
    Original paper: 'SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation,'
    https://arxiv.org/abs/1511.00561.
"""

__all__ = ['SegNet', 'segnet_cityscapes']

import os
import torch
import torch.nn as nn
from .common.conv import conv3x3, conv3x3_block
from .common.arch import DualPathSequential


class SegNet(nn.Module):
    """
    SegNet model from 'SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation,'
    https://arxiv.org/abs/1511.00561.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each stage in encoder and decoder.
    layers : list(list(int))
        Number of layers for each stage in encoder and decoder.
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
                 layers: list[list[int]],
                 aux: bool = False,
                 fixed_size: bool = False,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (1024, 2048),
                 num_classes: int = 19):
        super(SegNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        down_idx = 0
        up_idx = 1
        bias = True

        for i, out_channels in enumerate(channels[down_idx]):
            stage = nn.Sequential()
            for j in range(layers[down_idx][i]):
                if j < layers[down_idx][i] - 1:
                    unit = conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=bias)
                else:
                    unit = nn.MaxPool2d(
                        kernel_size=2,
                        stride=2,
                        return_indices=True)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            setattr(self, "down_stage{}".format(i + 1), stage)

        for i, channels_per_stage in enumerate(channels[up_idx]):
            stage = DualPathSequential(
                return_two=False,
                last_ordinals=(layers[up_idx][i] - 1),
                dual_path_scheme=(lambda module, x1, x2: (module(x1, x2), x2)))
            for j in range(layers[up_idx][i]):
                out_channels = in_channels if j < layers[up_idx][i] - 1 else channels_per_stage
                if j == 0:
                    unit = nn.MaxUnpool2d(
                        kernel_size=2,
                        stride=2)
                else:
                    unit = conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=bias)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            setattr(self, "up_stage{}".format(i + 1), stage)

        self.head = conv3x3(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x, max_indices1 = self.down_stage1(x)
        x, max_indices2 = self.down_stage2(x)
        x, max_indices3 = self.down_stage3(x)
        x, max_indices4 = self.down_stage4(x)
        x, max_indices5 = self.down_stage5(x)

        x = self.up_stage1(x, max_indices5)
        x = self.up_stage2(x, max_indices4)
        x = self.up_stage3(x, max_indices3)
        x = self.up_stage4(x, max_indices2)
        x = self.up_stage5(x, max_indices1)

        x = self.head(x)
        return x


def get_segnet(model_name: str | None = None,
               pretrained: bool = False,
               root: str = os.path.join("~", ".torch", "models"),
               **kwargs) -> nn.Module:
    """
    Create SegNet model with specific parameters.

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
    channels = [[64, 128, 256, 512, 512], [512, 256, 128, 64, 64]]
    layers = [[3, 3, 4, 4, 4], [4, 4, 4, 3, 2]]

    net = SegNet(
        channels=channels,
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


def segnet_cityscapes(num_classes: int = 19,
                      **kwargs) -> nn.Module:
    """
    SegNet model for Cityscapes from 'SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation,'
    https://arxiv.org/abs/1511.00561.

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
    return get_segnet(
        num_classes=num_classes,
        model_name="segnet_cityscapes",
        **kwargs)


def _test():
    from .common.model_store import calc_net_weight_count

    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        segnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != segnet_cityscapes or weight_count == 29453971)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
