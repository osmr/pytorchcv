"""
    Normalization common routines for models in PyTorch.
"""

__all__ = ['lambda_batchnorm1d', 'lambda_batchnorm2d', 'lambda_instancenorm2d', 'lambda_groupnorm',
           'create_normalization_layer', 'IBN']

import math
import torch
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


class IBN(nn.Module):
    """
    Instance-Batch Normalization block from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters
    ----------
    channels : int
        Number of channels.
    first_fraction : float, default 0.5
        The first fraction of channels for normalization.
    inst_first : bool, default True
        Whether instance normalization be on the first part of channels.
    """
    def __init__(self,
                 channels: int,
                 first_fraction: float = 0.5,
                 inst_first: bool = True):
        super(IBN, self).__init__()
        self.inst_first = inst_first
        h1_channels = int(math.floor(channels * first_fraction))
        h2_channels = channels - h1_channels
        self.split_sections = [h1_channels, h2_channels]

        if self.inst_first:
            self.inst_norm = nn.InstanceNorm2d(
                num_features=h1_channels,
                affine=True)
            self.batch_norm = nn.BatchNorm2d(num_features=h2_channels)
        else:
            self.batch_norm = nn.BatchNorm2d(num_features=h1_channels)
            self.inst_norm = nn.InstanceNorm2d(
                num_features=h2_channels,
                affine=True)

    def forward(self, x):
        x1, x2 = torch.split(x, split_size_or_sections=self.split_sections, dim=1)
        if self.inst_first:
            x1 = self.inst_norm(x1.contiguous())
            x2 = self.batch_norm(x2.contiguous())
        else:
            x1 = self.batch_norm(x1.contiguous())
            x2 = self.inst_norm(x2.contiguous())
        x = torch.cat((x1, x2), dim=1)
        return x
