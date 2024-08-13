"""
    VOCA for speech-driven facial animation, implemented in PyTorch.
    Original paper: 'Capture, Learning, and Synthesis of 3D Speaking Styles,' https://arxiv.org/abs/1905.03079.
"""

__all__ = ['VOCA', 'voca8flame']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common.conv import ConvBlock


class VocaEncoder(nn.Module):
    """
    VOCA encoder.

    Parameters
    ----------
    audio_features : int
        Number of audio features (characters/sounds).
    audio_window_size : int
        Size of audio window (for time related audio features).
    base_persons : int
        Number of base persons (subjects).
    encoder_features : int
        Number of encoder features.
    """
    def __init__(self,
                 audio_features: int,
                 audio_window_size: int,
                 base_persons: int,
                 encoder_features: int):
        super(VocaEncoder, self).__init__()
        self.audio_window_size = audio_window_size
        channels = (32, 32, 64, 64)
        fc1_channels = 128

        self.bn = nn.BatchNorm2d(num_features=1)

        in_channels = audio_features + base_persons
        self.branch = nn.Sequential()
        for i, out_channels in enumerate(channels):
            self.branch.add_module("conv{}".format(i + 1), ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
                bias=True,
                normalization=None))
            in_channels = out_channels

        in_channels += base_persons
        self.fc1 = nn.Linear(
            in_features=in_channels,
            out_features=fc1_channels)
        self.fc2 = nn.Linear(
            in_features=fc1_channels,
            out_features=encoder_features)

    def forward(self, x, pid):
        x = self.bn(x)
        x = x.transpose(1, 3).contiguous()
        y = pid.unsqueeze(-1).unsqueeze(-1)
        y = y.repeat(1, 1, self.audio_window_size, 1)
        x = torch.cat((x, y), dim=1)
        x = self.branch(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, pid), dim=1)
        x = self.fc1(x)
        x = x.tanh()
        x = self.fc2(x)
        return x


class VOCA(nn.Module):
    """
    VOCA model from 'Capture, Learning, and Synthesis of 3D Speaking Styles,' https://arxiv.org/abs/1905.03079.

    Parameters
    ----------
    audio_features : int, default 29
        Number of audio features (characters/sounds).
    audio_window_size : int, default 16
        Size of audio window (for time related audio features).
    base_persons : int, default 8
        Number of base persons (subjects).
    encoder_features : int, default 50
        Number of encoder features.
    vertices : int, default 5023
        Number of 3D geometry vertices.
    """
    def __init__(self,
                 audio_features: int = 29,
                 audio_window_size: int = 16,
                 base_persons: int = 8,
                 encoder_features: int = 50,
                 vertices: int = 5023):
        super(VOCA, self).__init__()
        self.base_persons = base_persons

        self.encoder = VocaEncoder(
            audio_features=audio_features,
            audio_window_size=audio_window_size,
            base_persons=base_persons,
            encoder_features=encoder_features)
        self.decoder = nn.Linear(
            in_features=encoder_features,
            out_features=(3 * vertices))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, pid):
        pid = F.one_hot(pid.long(), num_classes=self.base_persons).type_as(pid)
        x = self.encoder(x, pid)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, -1, 3)
        return x


def get_voca(base_persons: int,
             vertices: int,
             model_name: str | None = None,
             pretrained: bool = False,
             root: str = os.path.join("~", ".torch", "models"),
             **kwargs) -> nn.Module:
    """
    Create VOCA model with specific parameters.

    Parameters
    ----------
    base_persons : int
        Number of base persons (subjects).
    vertices : int
        Number of 3D geometry vertices.
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
    net = VOCA(
        base_persons=base_persons,
        vertices=vertices,
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


def voca8flame(**kwargs) -> nn.Module:
    """
    VOCA-8-FLAME model for 8 base persons and FLAME topology from 'Capture, Learning, and Synthesis of 3D Speaking
    Styles,' https://arxiv.org/abs/1905.03079.

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
    return get_voca(
        base_persons=8,
        vertices=5023,
        model_name="voca8flame",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        voca8flame,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != voca8flame or weight_count == 809563)

        batch = 14
        audio_features = 29
        audio_window_size = 16
        vertices = 5023

        x = torch.randn(batch, 1, audio_window_size, audio_features)
        pid = torch.full(size=(batch,), fill_value=3)
        y = net(x, pid)
        # y.sum().backward()
        assert (y.shape == (batch, 1, vertices, 3))


if __name__ == "__main__":
    _test()
