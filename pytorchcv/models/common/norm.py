"""
    Normalization common routines for models in PyTorch.
"""

__all__ = ['lambda_batchnorm1d', 'lambda_batchnorm2d', 'lambda_instancenorm2d', 'lambda_groupnorm',
           'create_normalization_layer']

from inspect import isfunction
import torch.nn as nn
from typing import Callable


def lambda_batchnorm1d(eps: float = 1e-5) -> Callable[[int], nn.Module]:
    """
    Create lambda-function generator for nn.BatchNorm1d normalization layer.

    Parameters
    ----------
    eps : float, default 1e-5
        Batch-norm epsilon.

    Returns
    -------
    function
        Desired function.
    """
    return lambda num_features: nn.BatchNorm1d(
        num_features=num_features,
        eps=eps)


def lambda_batchnorm2d(eps: float = 1e-5) -> Callable[[int], nn.Module]:
    """
    Create lambda-function generator for nn.BatchNorm2d normalization layer.

    Parameters
    ----------
    eps : float, default 1e-5
        Batch-norm epsilon.

    Returns
    -------
    function
        Desired function.
    """
    return lambda num_features: nn.BatchNorm2d(
        num_features=num_features,
        eps=eps)


def lambda_instancenorm2d(eps: float = 1e-5) -> Callable[[int], nn.Module]:
    """
    Create lambda-function generator for nn.InstanceNorm2d normalization layer.

    Parameters
    ----------
    eps : float, default 1e-5
        Instance-norm epsilon.

    Returns
    -------
    function
        Desired function.
    """
    return lambda num_features: nn.InstanceNorm2d(
        num_features=num_features,
        eps=eps)


def lambda_groupnorm(num_groups: int,
                     eps: float = 1e-5) -> Callable[[int], nn.Module]:
    """
    Create lambda-function generator for nn.GroupNorm normalization layer.

    Parameters
    ----------
    num_groups : int
        Group-norm number of groups.
    eps : float, default 1e-5
        Group-norm epsilon.

    Returns
    -------
    function
        Desired function.
    """
    return lambda num_features: nn.GroupNorm(
        num_groups=num_groups,
        num_channels=num_features,
        eps=eps)


def create_normalization_layer(normalization: Callable[..., nn.Module | None] | nn.Module,
                               **kwargs) -> nn.Module | None:
    """
    Create normalization layer from lambda-function generator or module.

    Parameters
    ----------
    normalization : function or nn.Module
        Lambda-function generator or module for normalization layer.

    Returns
    -------
    nn.Module or None
        Normalization layer.
    """
    assert (normalization is not None)
    if isfunction(normalization):
        return normalization(**kwargs)
    else:
        assert isinstance(normalization, nn.Module)
        return normalization
