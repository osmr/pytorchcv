"""
    Activation common routines for models in PyTorch.
"""

__all__ = ['Swish', 'HSigmoid', 'HSwish', 'lambda_relu', 'lambda_relu6', 'lambda_prelu', 'lambda_leakyrelu',
           'lambda_sigmoid', 'lambda_tanh', 'lambda_hsigmoid', 'lambda_swish', 'lambda_hswish',
           'create_activation_layer']

from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class Swish(nn.Module):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters
    ----------
    inplace : bool, default False
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace: bool = False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


def lambda_relu(inplace: bool = True) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.ReLU activation layer.

    Parameters
    ----------
    inplace : bool, default true
        Whether to do the operation in-place.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.ReLU(inplace=inplace)


def lambda_relu6(inplace: bool = True) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.ReLU6 activation layer.

    Parameters
    ----------
    inplace : bool, default true
        Whether to do the operation in-place.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.ReLU6(inplace=inplace)


def lambda_prelu(num_parameters: int = 1) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.PReLU activation layer.

    Parameters
    ----------
    num_parameters : int, default 1
        Number of `a` to learn. There is only two values are legitimate: 1, or the number of channels at input.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.PReLU(num_parameters=num_parameters)


def lambda_leakyrelu(negative_slope: float = 1e-2,
                     inplace: bool = True) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.LeakyReLU activation layer.

    Parameters
    ----------
    negative_slope : float, default 1e-2
        Slope coefficient controls the angle of the negative slope (which is used for negative input values).
    inplace : bool, default true
        Whether to do the operation in-place.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.LeakyReLU(
        negative_slope=negative_slope,
        inplace=inplace)


def lambda_sigmoid() -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.Sigmoid activation layer.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.Sigmoid()


def lambda_tanh() -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for nn.Tanh activation layer.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: nn.Tanh()


def lambda_hsigmoid() -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for HSigmoid activation layer.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: HSigmoid()


def lambda_swish() -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for Swish activation layer.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: Swish()


def lambda_hswish(inplace: bool = True) -> Callable[[], nn.Module]:
    """
    Create lambda-function generator for HSwish activation layer.

    Parameters
    ----------
    inplace : bool, default true
        Whether to do the operation in-place.

    Returns
    -------
    function
        Desired function.
    """
    return lambda: HSwish(inplace=inplace)


def create_activation_layer(activation: Callable[..., nn.Module | None] | nn.Module | str) -> nn.Module | None:
    """
    Create activation layer from lambda-function generator or module.

    Parameters
    ----------
    activation : function or nn.Module or str
        Lambda-function generator or module for activation layer.

    Returns
    -------
    nn.Module or None
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "swish":
            return Swish()
        elif activation == "hswish":
            return HSwish(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "hsigmoid":
            return HSigmoid()
        else:
            raise NotImplementedError()
    else:
        assert isinstance(activation, nn.Module)
        return activation
