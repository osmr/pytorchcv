"""
    ProPainter (Image Propagation), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['PPImagePropagation', 'propainter_ip', 'BidirectionalPropagation']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common.activ import lambda_leakyrelu
from .resnet import ResBlock
from .propainter_rfc import SecondOrderDeformableAlignment


def flow_warp(x,
              flow,
              interpolation="bilinear",
              padding_mode="zeros",
              align_corners=True):
    """
    Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f"The spatial sizes of input ({x.size()[-2:]}) and "
                         f"flow ({flow.size()[1:3]}) are not the same.")
    _, _, height, width = x.size()
    # Create mesh grid:
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, height, device=device),
        torch.arange(0, width, device=device),
        indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(width - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(height - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        input=x,
        grid=grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fb_consistency_check(flow_fw,
                         flow_bw,
                         alpha1=0.01,
                         alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw


class BidirectionalPropagation(nn.Module):
    def __init__(self,
                 channels: int,
                 learnable: bool = True):
        super(BidirectionalPropagation, self).__init__()
        self.channels = channels
        self.learnable = learnable

        activation = lambda_leakyrelu(negative_slope=0.2)
        self.prop_list = ["backward_1", "forward_1"]

        if self.learnable:
            self.deform_align = nn.ModuleDict()
            self.backbone = nn.ModuleDict()

            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    x_in_channels=channels,
                    cond_in_channels=(2 * channels + 2 + 1 + 2),
                    out_channels=channels,
                    deform_groups=16,
                    max_residue_magnitude=3)
                self.backbone[module] = ResBlock(
                    in_channels=(2 * channels + 2),
                    out_channels=channels,
                    stride=1,
                    bias=True,
                    normalization=None,
                    activation=activation)

            self.fuse = ResBlock(
                in_channels=(2 * channels + 2),
                out_channels=channels,
                stride=1,
                bias=True,
                normalization=None,
                activation=activation)

    def binary_mask(self,
                    mask,
                    th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        # return mask.float()
        return mask.to(mask)

    def forward(self,
                x,
                flows_forward,
                flows_backward,
                mask,
                interpolation="bilinear"):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        batch, time, channels, height, width = x.shape
        assert (channels == self.channels)

        feats = {}
        masks = {}
        feats["input"] = [x[:, i, :, :, :] for i in range(0, time)]
        masks["input"] = [mask[:, i, :, :, :] for i in range(0, time)]

        prop_list = ["backward_1", "forward_1"]
        cache_list = ["input"] + prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if "backward" in module_name:
                frame_idx = range(0, time)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, time)
                flow_idx = range(-1, time - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = fb_consistency_check(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([
                            feat_current,
                            feat_warped,
                            flow_prop,
                            flow_vaild_mask,
                            mask_current], dim=1)
                        feat_prop = self.deform_align[module_name](x=feat_prop, cond=cond, flow=flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                        mask_prop_valid = self.binary_mask(mask_prop_valid)

                        union_vaild_mask = self.binary_mask(mask_current * flow_vaild_mask * (1 - mask_prop_valid))
                        feat_prop = union_vaild_mask * feat_warped + (1 - union_vaild_mask) * feat_current
                        # update mask
                        mask_prop = self.binary_mask(mask_current * (1 - (flow_vaild_mask * (1 - mask_prop_valid))))

                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(feats["backward_1"], dim=1).view(-1, channels, height, width)
        outputs_f = torch.stack(feats["forward_1"], dim=1).view(-1, channels, height, width)

        if self.learnable:
            mask_in = mask.view(-1, 2, height, width)
            masks_f = None
            outputs = (self.fuse(torch.cat([outputs_b, outputs_f, mask_in], dim=1)) +
                       x.view(-1, channels, height, width))
        else:
            masks_f = torch.stack(masks["forward_1"], dim=1)
            outputs = outputs_f

        return (outputs_b.view(batch, -1, channels, height, width),
                outputs_f.view(batch, -1, channels, height, width),
                outputs.view(batch, -1, channels, height, width),
                masks_f)


class PPImagePropagation(BidirectionalPropagation):
    """
    Image Propagation part of ProPainter model from 'ProPainter: Improving Propagation and Transformer for Video
    Inpainting,' https://arxiv.org/pdf/2309.03897.

    Parameters
    ----------
    in_channels : int, default 3
        Number of input channels.
    """
    def __init__(self,
                 in_channels: int = 3):
        super(PPImagePropagation, self).__init__(
            channels=in_channels,
            learnable=False)

    def forward(self,
                frames: torch.Tensor,
                masks: torch.Tensor,
                comp_flows: torch.Tensor,
                interpolation: str = "nearest"):
        """
        Do image propagation.
        Batch dimension for frames is interpreted as time.

        Parameters
        ----------
        frames : torch.Tensor
            Frames with size: (time, channels=3, height, width).
        masks : torch.Tensor
            Masks with size: (time, channels=1, height, width).
        comp_flows : torch.Tensor
            Bidirectional completed flows with size: (time-1, channels=4, height, width).
        interpolation : str, default 'nearest'
            Interpolation mode.

        Returns
        -------
        torch.Tensor
            Updated frames with size: (time, channels=4, height, width).
        torch.Tensor
            Updated masks with size: (time, channels=1, height, width).
        """
        assert (len(frames.shape) == 4)
        assert (len(masks.shape) == 4)
        assert (len(comp_flows.shape) == 4)
        assert (frames.shape[0] == masks.shape[0])
        assert (frames.shape[0] > 1)
        assert (comp_flows.shape[0] == masks.shape[0] - 1)
        assert (frames.shape[1] == 3)
        assert (masks.shape[1] == 1)
        assert (comp_flows.shape[1] == 4)
        assert (frames.shape[2] == masks.shape[2])
        assert (frames.shape[3] == masks.shape[3])
        assert (frames.shape[2] == comp_flows.shape[2])
        assert (frames.shape[3] == comp_flows.shape[3])

        masked_frames = frames * (1 - masks)
        comp_flows_forward, comp_flows_backward = torch.split(comp_flows, [2, 2], dim=1)

        _, _, prop_frames, updated_masks = super(PPImagePropagation, self).forward(
            x=masked_frames[None],
            flows_forward=comp_flows_forward[None],
            flows_backward=comp_flows_backward[None],
            mask=masks[None],
            interpolation=interpolation)

        prop_frames = prop_frames[0]
        updated_masks = updated_masks[0]

        return prop_frames, updated_masks


def get_propainter_ip(model_name: str | None = None,
                      pretrained: bool = False,
                      root: str = os.path.join("~", ".torch", "models"),
                      **kwargs) -> nn.Module:
    """
    Create Recurrent Image Propagation part of ProPainter model with specific parameters.

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
    net = PPImagePropagation(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .common.model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def propainter_ip(**kwargs) -> nn.Module:
    """
    ProPainter-IP model from 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.

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
    return get_propainter_ip(
        model_name="propainter_ip",
        **kwargs)


def _test():
    import torch
    from .common.model_store import calc_net_weight_count

    pretrained = False

    models = [
        propainter_ip,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weight_count(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != propainter_ip or weight_count == 0)

        # batch = 4
        time = 5
        height = 240
        width = 432
        frames = torch.randn(time, 3, height, width)
        masks = torch.randn(time, 1, height, width)
        comp_flows = torch.randn(time - 1, 4, height, width)

        prop_frames, updated_masks = net(
            frames=frames,
            masks=masks,
            comp_flows=comp_flows,
            interpolation="nearest")
        # y1.sum().backward()
        # y2.sum().backward()
        assert (tuple(prop_frames.size()) == (time, 3, height, width))
        assert (tuple(updated_masks.size()) == (time, 1, height, width))


if __name__ == "__main__":
    _test()
