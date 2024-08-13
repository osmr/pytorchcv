"""
    ProxylessNAS for CUB-200-2011, implemented in Gluon.
    Original paper: 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.
"""

__all__ = ['proxylessnas_cpu_cub', 'proxylessnas_gpu_cub', 'proxylessnas_mobile_cub', 'proxylessnas_mobile14_cub']

import torch.nn as nn
from .proxylessnas import get_proxylessnas


def proxylessnas_cpu_cub(num_classes: int = 200,
                         **kwargs) -> nn.Module:
    """
    ProxylessNAS (CPU) model for CUB-200-2011 from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and
    Hardware,' https://arxiv.org/abs/1812.00332.

    Parameters
    ----------
    num_classes : int, default 200
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
    return get_proxylessnas(
        num_classes=num_classes,
        version="cpu",
        model_name="proxylessnas_cpu_cub",
        **kwargs)


def proxylessnas_gpu_cub(num_classes: int = 200,
                         **kwargs) -> nn.Module:
    """
    ProxylessNAS (GPU) model for CUB-200-2011 from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and
    Hardware,' https://arxiv.org/abs/1812.00332.

    Parameters
    ----------
    num_classes : int, default 200
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
    return get_proxylessnas(
        num_classes=num_classes,
        version="gpu",
        model_name="proxylessnas_gpu_cub",
        **kwargs)


def proxylessnas_mobile_cub(num_classes: int = 200,
                            **kwargs) -> nn.Module:
    """
    ProxylessNAS (Mobile) model for CUB-200-2011 from 'ProxylessNAS: Direct Neural Architecture Search on Target Task
    and Hardware,' https://arxiv.org/abs/1812.00332.

    Parameters
    ----------
    num_classes : int, default 200
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
    return get_proxylessnas(
        num_classes=num_classes,
        version="mobile",
        model_name="proxylessnas_mobile_cub",
        **kwargs)


def proxylessnas_mobile14_cub(num_classes: int = 200,
                              **kwargs) -> nn.Module:
    """
    ProxylessNAS (Mobile-14) model for CUB-200-2011 from 'ProxylessNAS: Direct Neural Architecture Search on Target Task
    and Hardware,' https://arxiv.org/abs/1812.00332.

    Parameters
    ----------
    num_classes : int, default 200
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
    return get_proxylessnas(
        num_classes=num_classes,
        version="mobile14",
        model_name="proxylessnas_mobile14_cub",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        proxylessnas_cpu_cub,
        proxylessnas_gpu_cub,
        proxylessnas_mobile_cub,
        proxylessnas_mobile14_cub,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != proxylessnas_cpu_cub or weight_count == 3215248)
        assert (model != proxylessnas_gpu_cub or weight_count == 5736648)
        assert (model != proxylessnas_mobile_cub or weight_count == 3055712)
        assert (model != proxylessnas_mobile14_cub or weight_count == 5423168)

        x = torch.randn(14, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, 200))


if __name__ == "__main__":
    _test()
