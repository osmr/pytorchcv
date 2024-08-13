"""
    SimplePose for COCO Keypoint, implemented in PyTorch.
    Original paper: 'Simple Baselines for Human Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.
"""

__all__ = ['SimplePose', 'simplepose_resnet18_coco', 'simplepose_resnet50b_coco', 'simplepose_resnet101b_coco',
           'simplepose_resnet152b_coco', 'simplepose_resneta50b_coco', 'simplepose_resneta101b_coco',
           'simplepose_resneta152b_coco']

import os
import torch
import torch.nn as nn
from .common.conv import DeconvBlock, conv1x1
from .common.tutti import HeatmapMaxDetBlock
from .resnet import resnet18, resnet50b, resnet101b, resnet152b
from .resneta import resneta50b, resneta101b, resneta152b


class SimplePose(nn.Module):
    """
    SimplePose model from 'Simple Baselines for Human Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels for the backbone.
    channels : list(int)
        Number of output channels for each decoder unit.
    return_heatmap : bool, default False
        Whether to return only heatmap.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (256, 192)
        Spatial size of the expected input image.
    keypoints : int, default 17
        Number of keypoints.
    """
    def __init__(self,
                 backbone: nn.Sequential,
                 backbone_out_channels: int,
                 channels: list[int],
                 return_heatmap: bool = False,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (256, 192),
                 keypoints: int = 17):
        super(SimplePose, self).__init__()
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap

        self.backbone = backbone

        self.decoder = nn.Sequential()
        in_channels = backbone_out_channels
        for i, out_channels in enumerate(channels):
            self.decoder.add_module("unit{}".format(i + 1), DeconvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1))
            in_channels = out_channels
        self.decoder.add_module("final_block", conv1x1(
            in_channels=in_channels,
            out_channels=keypoints,
            bias=True))

        self.heatmap_max_det = HeatmapMaxDetBlock()

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        heatmap = self.decoder(x)
        if self.return_heatmap:
            return heatmap
        else:
            keypoints = self.heatmap_max_det(heatmap)
            return keypoints


def get_simplepose(backbone: nn.Sequential,
                   backbone_out_channels: int,
                   keypoints: int,
                   model_name: str | None = None,
                   pretrained: bool = False,
                   root: str = os.path.join("~", ".torch", "models"),
                   **kwargs) -> nn.Module:
    """
    Create SimplePose model with specific parameters.

    Parameters
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels for the backbone.
    keypoints : int
        Number of keypoints.
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
    channels = [256, 256, 256]

    net = SimplePose(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
        keypoints=keypoints,
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


def simplepose_resnet18_coco(pretrained_backbone: bool = False,
                             keypoints: int = 17,
                             **kwargs) -> nn.Module:
    """
    SimplePose model on the base of ResNet-18 for COCO Keypoint from 'Simple Baselines for Human Pose Estimation and
    Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(
        backbone=backbone,
        backbone_out_channels=512,
        keypoints=keypoints,
        model_name="simplepose_resnet18_coco",
        **kwargs)


def simplepose_resnet50b_coco(pretrained_backbone: bool = False,
                              keypoints: int = 17,
                              **kwargs) -> nn.Module:
    """
    SimplePose model on the base of ResNet-50b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation and
    Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(
        backbone=backbone,
        backbone_out_channels=2048,
        keypoints=keypoints,
        model_name="simplepose_resnet50b_coco",
        **kwargs)


def simplepose_resnet101b_coco(pretrained_backbone: bool = False,
                               keypoints: int = 17,
                               **kwargs) -> nn.Module:
    """
    SimplePose model on the base of ResNet-101b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(
        backbone=backbone,
        backbone_out_channels=2048,
        keypoints=keypoints,
        model_name="simplepose_resnet101b_coco",
        **kwargs)


def simplepose_resnet152b_coco(pretrained_backbone: bool = False,
                               keypoints: int = 17,
                               **kwargs) -> nn.Module:
    """
    SimplePose model on the base of ResNet-152b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    backbone = resnet152b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(
        backbone=backbone,
        backbone_out_channels=2048,
        keypoints=keypoints,
        model_name="simplepose_resnet152b_coco",
        **kwargs)


def simplepose_resneta50b_coco(pretrained_backbone: bool = False,
                               keypoints: int = 17,
                               **kwargs) -> nn.Module:
    """
    SimplePose model on the base of ResNet(A)-50b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    backbone = resneta50b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(
        backbone=backbone,
        backbone_out_channels=2048,
        keypoints=keypoints,
        model_name="simplepose_resneta50b_coco",
        **kwargs)


def simplepose_resneta101b_coco(pretrained_backbone: bool = False,
                                keypoints: int = 17,
                                **kwargs) -> nn.Module:
    """
    SimplePose model on the base of ResNet(A)-101b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    backbone = resneta101b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(
        backbone=backbone,
        backbone_out_channels=2048,
        keypoints=keypoints,
        model_name="simplepose_resneta101b_coco",
        **kwargs)


def simplepose_resneta152b_coco(pretrained_backbone: bool = False,
                                keypoints: int = 17,
                                **kwargs) -> nn.Module:
    """
    SimplePose model on the base of ResNet(A)-152b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    backbone = resneta152b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(
        backbone=backbone,
        backbone_out_channels=2048,
        keypoints=keypoints,
        model_name="simplepose_resneta152b_coco",
        **kwargs)


def _test():
    from .common.model_store import calc_net_weight_count

    in_size = (256, 192)
    keypoints = 17
    return_heatmap = False
    pretrained = False

    models = [
        simplepose_resnet18_coco,
        simplepose_resnet50b_coco,
        simplepose_resnet101b_coco,
        simplepose_resnet152b_coco,
        simplepose_resneta50b_coco,
        simplepose_resneta101b_coco,
        simplepose_resneta152b_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != simplepose_resnet18_coco or weight_count == 15376721)
        assert (model != simplepose_resnet50b_coco or weight_count == 33999697)
        assert (model != simplepose_resnet101b_coco or weight_count == 52991825)
        assert (model != simplepose_resnet152b_coco or weight_count == 68635473)
        assert (model != simplepose_resneta50b_coco or weight_count == 34018929)
        assert (model != simplepose_resneta101b_coco or weight_count == 53011057)
        assert (model != simplepose_resneta152b_coco or weight_count == 68654705)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        assert ((y.shape[0] == batch) and (y.shape[1] == keypoints))
        if return_heatmap:
            assert ((y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4))
        else:
            assert (y.shape[2] == 3)


if __name__ == "__main__":
    _test()
