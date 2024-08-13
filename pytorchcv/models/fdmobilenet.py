"""
    FD-MobileNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.
"""

__all__ = ['get_fdmobilenet', 'fdmobilenet_w1', 'fdmobilenet_w3d4', 'fdmobilenet_wd2', 'fdmobilenet_wd4']

import os
import torch.nn as nn
from .mobilenet import MobileNet


def get_fdmobilenet(width_scale: float,
                    model_name: str | None = None,
                    pretrained: bool = False,
                    root: str = os.path.join("~", ".torch", "models"),
                    **kwargs) -> nn.Module:
    """
    Create FD-MobileNet model with specific parameters.

    Parameters
    ----------
    width_scale : float
        Scale factor for width of layers.
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
    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 1024]]
    first_stage_stride = True

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride,
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


def fdmobilenet_w1(**kwargs) -> nn.Module:
    """
    FD-MobileNet 1.0x model from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

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
    return get_fdmobilenet(
        width_scale=1.0,
        model_name="fdmobilenet_w1",
        **kwargs)


def fdmobilenet_w3d4(**kwargs) -> nn.Module:
    """
    FD-MobileNet 0.75x model from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

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
    return get_fdmobilenet(
        width_scale=0.75,
        model_name="fdmobilenet_w3d4",
        **kwargs)


def fdmobilenet_wd2(**kwargs) -> nn.Module:
    """
    FD-MobileNet 0.5x model from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

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
    return get_fdmobilenet(
        width_scale=0.5,
        model_name="fdmobilenet_wd2",
        **kwargs)


def fdmobilenet_wd4(**kwargs) -> nn.Module:
    """
    FD-MobileNet 0.25x model from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

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
    return get_fdmobilenet(
        width_scale=0.25,
        model_name="fdmobilenet_wd4",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        fdmobilenet_w1,
        fdmobilenet_w3d4,
        fdmobilenet_wd2,
        fdmobilenet_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fdmobilenet_w1 or weight_count == 2901288)
        assert (model != fdmobilenet_w3d4 or weight_count == 1833304)
        assert (model != fdmobilenet_wd2 or weight_count == 993928)
        assert (model != fdmobilenet_wd4 or weight_count == 383160)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
