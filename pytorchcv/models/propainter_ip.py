"""
    ProPainter (Image Propagation), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import lambda_leakyrelu
from resnet import ResBlock
from propainter_rfc import SecondOrderDeformableAlignment


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
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw,
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

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()

        if self.learnable:
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
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
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


class InpaintGenerator(nn.Module):
    def __init__(self):
        super(InpaintGenerator, self).__init__()

        self.img_prop_module = BidirectionalPropagation(3, learnable=False)

        self._init_weights()

    def _init_weights(self,
                      init_type="normal",
                      gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("InstanceNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def forward(self,
                masked_frames,
                completed_flows,
                masks,
                interpolation="nearest"):
        _, _, prop_frames, updated_masks = self.img_prop_module(
            x=masked_frames,
            flows_forward=completed_flows[0],
            flows_backward=completed_flows[1],
            mask=masks,
            interpolation=interpolation)
        return prop_frames, updated_masks


def _test2():
    import re
    import os
    import numpy as np

    def convert_state_dict(src_checkpoint,
                           dst_checkpoint):

        src_param_keys = list(src_checkpoint.keys())

        upd_dict = {}

        list2 = list(filter(re.compile("feat_prop_module.deform_align.").search, src_param_keys))
        list2_u = [key.replace(".backward_1.weight", ".backward_1.deform_conv.weight") for key in list2]
        list2_u = [key.replace(".backward_1.bias", ".backward_1.deform_conv.bias") for key in list2_u]
        list2_u = [key.replace(".forward_1.weight", ".forward_1.deform_conv.weight") for key in list2_u]
        list2_u = [key.replace(".forward_1.bias", ".forward_1.deform_conv.bias") for key in list2_u]
        list2_u = [key.replace(".conv_offset.0.", ".conv_offset.conv1.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.2.", ".conv_offset.conv2.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.4.", ".conv_offset.conv3.conv.") for key in list2_u]
        list2_u = [key.replace(".conv_offset.6.", ".conv_offset.conv4.conv.") for key in list2_u]
        for src_i, dst_i in zip(list2, list2_u):
            upd_dict[src_i] = dst_i

        list3 = list(filter(re.compile("feat_prop_module.fuse.").search, src_param_keys))
        list3_u = [key.replace(".0.", ".conv1.conv.") for key in list3]
        list3_u = [key.replace(".2.", ".conv2.conv.") for key in list3_u]
        for src_i, dst_i in zip(list3, list3_u):
            upd_dict[src_i] = dst_i

        list8 = list(filter(re.compile("feat_prop_module.backbone.").search, src_param_keys))
        list8_u = [key.replace(".0.", ".conv1.conv.") for key in list8]
        list8_u = [key.replace(".2.", ".conv2.conv.") for key in list8_u]
        for src_i, dst_i in zip(list8, list8_u):
            upd_dict[src_i] = dst_i

        list4_r = []

        for k, v in src_checkpoint.items():
            if k in upd_dict.keys():
                dst_checkpoint[upd_dict[k]] = src_checkpoint[k]
            else:
                if k not in list4_r:
                    dst_checkpoint[k] = src_checkpoint[k]
                else:
                    print("Remove: {}".format(k))
                    pass

    root_path = "../../../pytorchcv_data/test_a"
    pp_model_file_name = "ProPainter.pth"

    model_path = os.path.join(root_path, pp_model_file_name)
    net_pp = InpaintGenerator()

    src_checkpoint = torch.load(model_path, map_location="cpu")
    dst_checkpoint = net_pp.state_dict()
    convert_state_dict(
        src_checkpoint,
        dst_checkpoint)
    net_pp.load_state_dict(dst_checkpoint, strict=True)
    # ckpt = torch.load(model_path, map_location="cpu")
    # net_pp.load_state_dict(ckpt, strict=True)

    for p in net_pp.parameters():
        p.requires_grad = False
    net_pp.eval()
    net_pp = net_pp.cuda()

    frame1_file_path = os.path.join(root_path, "frame_00100.npy")
    frame2_file_path = os.path.join(root_path, "frame_00101.npy")
    frame1_np = np.load(frame1_file_path)
    frame2_np = np.load(frame2_file_path)

    mask_dilated1_file_path = os.path.join(root_path, "mask_dilated_00100.npy")
    mask_dilated2_file_path = os.path.join(root_path, "mask_dilated_00101.npy")
    mask_dilated1_np = np.load(mask_dilated1_file_path)
    mask_dilated2_np = np.load(mask_dilated2_file_path)

    pred_flow_f_file_path = os.path.join(root_path, "pred_flow_f_00100.npy")
    pred_flow_b_file_path = os.path.join(root_path, "pred_flow_b_00100.npy")
    pred_flow_f_np = np.load(pred_flow_f_file_path)
    pred_flow_b_np = np.load(pred_flow_b_file_path)

    frames_np = np.stack([frame1_np, frame2_np])[None]
    frames = torch.from_numpy(frames_np).cuda()
    masks_dilated_np = np.stack([mask_dilated1_np, mask_dilated2_np])[None]
    masks_dilated = torch.from_numpy(masks_dilated_np).cuda()
    masked_frames = frames * (1 - masks_dilated)

    pred_flow_f_np = pred_flow_f_np[None, None]
    pred_flow_b_np = pred_flow_b_np[None, None]
    pred_flow_f = torch.from_numpy(pred_flow_f_np).cuda()
    pred_flow_b = torch.from_numpy(pred_flow_b_np).cuda()

    prop_imgs, updated_local_masks = net_pp.img_propagation(
        masked_frames=masked_frames,
        completed_flows=(pred_flow_f, pred_flow_b),
        masks=masks_dilated,
        interpolation="nearest")
    pred_img = net_pp(
        masked_frames=masked_frames,
        completed_flows=(pred_flow_f, pred_flow_b),
        masks_in=masks_dilated,
        masks_updated=masks_dilated,
        num_local_frames=2)

    pred_img_np_ = pred_img[0].cpu().detach().numpy()

    # np.save(os.path.join(root_path, "pred_img.npy"), np.ascontiguousarray(pred_img_np_))

    pred_img_file_path = os.path.join(root_path, "pred_img.npy")
    pred_img_np = np.load(pred_img_file_path)

    if not np.array_equal(pred_img_np, pred_img_np_):
        print("*")
    np.testing.assert_array_equal(pred_img_np, pred_img_np_)

    pass


# def _test():
#     import torch
#     from model_store import calc_net_weight_count
#
#     pretrained = False
#
#     models = [
#         raft_things,
#         raft_small,
#     ]
#
#     for model in models:
#
#         net = model(pretrained=pretrained)
#
#         # net.train()
#         net.eval()
#         weight_count = calc_net_weight_count(net)
#         print("m={}, {}".format(model.__name__, weight_count))
#         assert (model != raft_things or weight_count == 5257536)
#         assert (model != raft_small or weight_count == 990162)
#
#         batch = 4
#         height = 240
#         width = 432
#         x1 = torch.randn(batch, 3, height, width)
#         x2 = torch.randn(batch, 3, height, width)
#         y1, y2 = net(x1, x2)
#         # y1.sum().backward()
#         # y2.sum().backward()
#         assert (tuple(y1.size()) == (batch, 2, height // 8, width // 8))
#         assert (tuple(y2.size()) == (batch, 2, height, width))


if __name__ == "__main__":
    _test2()
