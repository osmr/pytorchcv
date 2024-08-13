"""
    PFPCNet for 3D face reconstruction, implemented in PyTorch.
    Original paper: 'Production-Level Facial Performance Capture Using Deep Convolutional Neural Networks,'
    https://arxiv.org/abs/1609.06536.
"""

__all__ = ['PFPCNet', 'pfpcnet']

import os
import torch.nn as nn
from .common.norm import lambda_batchnorm2d
from .common.conv import conv3x3_block


class PFPCNet(nn.Module):
    """
    PFPCNet model from 'Production-Level Facial Performance Capture Using Deep Convolutional Neural Networks,'
    https://arxiv.org/abs/1609.06536.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    pca_size : int
        Number of PCA coefficients (number of blendshapes).
    use_bn : bool, default False
        Whether to use BatchNorm layers.
    in_channels : int, default 1
        Number of input channels.
    in_size : tuple(int, int), default (320, 240)
        Spatial size of the expected input image.
    vertices : int, default 5023
        Number of 3D geometry vertices.
    """
    def __init__(self,
                 channels: list[list[int]],
                 pca_size: int,
                 use_bn: bool = True,
                 in_channels: int = 1,
                 in_size: tuple[int, int] = (320, 240),
                 vertices: int = 5023):
        super(PFPCNet, self).__init__()
        self.in_size = in_size
        self.vertices = vertices
        normalization = lambda_batchnorm2d() if use_bn else None

        self.encoder = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if j == 0 else 1
                stage.add_module("unit{}".format(j + 1), conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    normalization=normalization))
                in_channels = out_channels
            self.encoder.add_module("stage{}".format(i + 1), stage)

        self.decoder = nn.Sequential()
        self.decoder.add_module("dropout", nn.Dropout(p=0.2))
        self.decoder.add_module("fc1", nn.Linear(
            in_features=(in_channels * 5 * 4),
            out_features=pca_size))
        self.decoder.add_module("fc2", nn.Linear(
            in_features=pca_size,
            out_features=(3 * vertices)))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        x = x.view(x.size(0), -1, 3)
        return x


def get_pfpcnet(model_name: str | None = None,
                pretrained: bool = False,
                root: str = os.path.join("~", ".torch", "models"),
                **kwargs) -> nn.Module:
    """
    Create PFPCNet model with specific parameters.

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
    layers = [2, 2, 2, 2, 2, 2]
    channels_per_layers = [64, 96, 144, 216, 324, 486]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    pca_size = 120

    net = PFPCNet(
        channels=channels,
        pca_size=pca_size,
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


def pfpcnet(**kwargs) -> nn.Module:
    """
    PFPCNet model from 'Production-Level Facial Performance Capture Using Deep Convolutional Neural Networks,'
    https://arxiv.org/abs/1609.06536.

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
    return get_pfpcnet(
        model_name="pfpcnet",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        pfpcnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != pfpcnet or weight_count == 9299329)

        batch = 4
        in_channels = 1
        vertices = 5023
        x = torch.randn(batch, in_channels, 320, 240)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (batch, vertices, 3))


if __name__ == "__main__":
    _test()
