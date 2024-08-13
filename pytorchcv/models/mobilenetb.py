"""
    MobileNet(B) with simplified depthwise separable convolution block for ImageNet-1K, implemented in Gluon.
    Original paper: 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.
"""

__all__ = ['mobilenetb_w1', 'mobilenetb_w3d4', 'mobilenetb_wd2', 'mobilenetb_wd4']

import torch.nn as nn
from .mobilenet import get_mobilenet


def mobilenetb_w1(**kwargs) -> nn.Module:
    """
    1.0 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.

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
    return get_mobilenet(
        width_scale=1.0,
        dws_simplified=True,
        model_name="mobilenetb_w1",
        **kwargs)


def mobilenetb_w3d4(**kwargs) -> nn.Module:
    """
    0.75 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.

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
    return get_mobilenet(
        width_scale=0.75,
        dws_simplified=True,
        model_name="mobilenetb_w3d4",
        **kwargs)


def mobilenetb_wd2(**kwargs) -> nn.Module:
    """
    0.5 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.

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
    return get_mobilenet(
        width_scale=0.5,
        dws_simplified=True,
        model_name="mobilenetb_wd2",
        **kwargs)


def mobilenetb_wd4(**kwargs) -> nn.Module:
    """
    0.25 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.

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
    return get_mobilenet(
        width_scale=0.25,
        dws_simplified=True,
        model_name="mobilenetb_wd4",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        mobilenetb_w1,
        mobilenetb_w3d4,
        mobilenetb_wd2,
        mobilenetb_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenetb_w1 or weight_count == 4222056)
        assert (model != mobilenetb_w3d4 or weight_count == 2578120)
        assert (model != mobilenetb_wd2 or weight_count == 1326632)
        assert (model != mobilenetb_wd4 or weight_count == 467592)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
