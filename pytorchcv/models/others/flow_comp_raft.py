import os
import torch
import torch.nn as nn
import numpy as np
import re

from raft0 import RAFT0
from raft import RAFT


def convert_state_dict(src_checkpoint,
                       dst_checkpoint,
                       small=False):

    src_param_keys = list(src_checkpoint.keys())

    upd_dict = {}

    list1 = list(filter(re.compile("net.conv1.").search, src_param_keys))
    list1_u = [key.replace("net.conv1.", "net.features.init_block.conv.") for key in list1]
    for src_i, dst_i in zip(list1, list1_u):
        upd_dict[src_i] = dst_i

    list2 = list(filter(re.compile("net.conv2.").search, src_param_keys))
    list2_u = [key.replace("net.conv2.", "net.features.final_block.") for key in list2]
    for src_i, dst_i in zip(list2, list2_u):
        upd_dict[src_i] = dst_i

    list3 = list(filter(re.compile("net.norm1.").search, src_param_keys))
    list3_u = [key.replace("net.norm1.", "net.features.init_block.bn.") for key in list3]
    for src_i, dst_i in zip(list3, list3_u):
        upd_dict[src_i] = dst_i

    list5 = list(filter(re.compile("update_block.encoder.").search, src_param_keys))
    list5_u = [key.replace(".conv.", ".conv_out.conv.") for key in list5]
    if small:
        list5_u = [key.replace(".convc1.", ".conv_corr.conv.") for key in list5_u]
    else:
        list5_u = [key.replace(".convc1.", ".conv_corr.conv_list.conv1.conv.") for key in list5_u]
        list5_u = [key.replace(".convc2.", ".conv_corr.conv_list.conv2.conv.") for key in list5_u]
    list5_u = [key.replace(".convf1.", ".conv_flow.conv_list.conv1.conv.") for key in list5_u]
    list5_u = [key.replace(".convf2.", ".conv_flow.conv_list.conv2.conv.") for key in list5_u]
    for src_i, dst_i in zip(list5, list5_u):
        upd_dict[src_i] = dst_i

    list4 = list(filter(re.compile("net.layer").search, src_param_keys))
    list4_u = [key.replace("net.layer1.0.", "net.features.stage1.unit1.body.") for key in list4]
    list4_u = [key.replace("net.layer1.1.", "net.features.stage1.unit2.body.") for key in list4_u]
    list4_u = [key.replace("net.layer2.0.", "net.features.stage2.unit1.body.") for key in list4_u]
    list4_u = [key.replace("net.layer2.1.", "net.features.stage2.unit2.body.") for key in list4_u]
    list4_u = [key.replace("net.layer3.0.", "net.features.stage3.unit1.body.") for key in list4_u]
    list4_u = [key.replace("net.layer3.1.", "net.features.stage3.unit2.body.") for key in list4_u]
    list4_u = [key.replace(".body.downsample.0.", ".identity_conv.conv.") for key in list4_u]
    list4_u = [key.replace(".body.downsample.1.", ".identity_conv.bn.") for key in list4_u]
    list4_u = [key.replace(".conv1.", ".conv1.conv.") for key in list4_u]
    list4_u = [key.replace(".conv2.", ".conv2.conv.") for key in list4_u]
    if small:
        list4_u = [key.replace(".conv3.", ".conv3.conv.") for key in list4_u]
    list4_u = [key.replace(".norm1.", ".conv1.bn.") for key in list4_u]
    list4_u = [key.replace(".norm2.", ".conv2.bn.") for key in list4_u]

    if small:
        list4_r = list(filter(re.compile(".norm4.").search, list4))
    else:
        list4_r = list(filter(re.compile(".norm3.").search, list4))

    for src_i, dst_i in zip(list4, list4_u):
        if src_i not in list4_r:
            upd_dict[src_i] = dst_i

    for k, v in src_checkpoint.items():
        if k in upd_dict.keys():
            dst_checkpoint[upd_dict[k]] = src_checkpoint[k]
        else:
            if k not in list4_r:
                dst_checkpoint[k] = src_checkpoint[k]
            else:
                print("Remove: {}".format(k))
                pass


def initialize_RAFT(model_path='weights/raft-things.pth',
                    device='cuda',
                    small=False,
                    raft_orig=False):
    """Initializes the RAFT model.
    """
    if raft_orig:
        net = RAFT0(small=small)
    else:
        net = RAFT(small=small)
    src_checkpoint = torch.load(model_path, map_location="cpu")

    # net_tmp = torch.nn.DataParallel(net)
    # net_tmp.load_state_dict(checkpoint)
    # checkpoint = net_tmp.module.cpu().state_dict()

    if raft_orig:
        dst_checkpoint = src_checkpoint
    else:
        dst_checkpoint = net.state_dict()
        convert_state_dict(
            src_checkpoint,
            dst_checkpoint,
            small)

    net.load_state_dict(dst_checkpoint)

    net.to(device)

    return net


class RAFT_bi(nn.Module):
    """Flow completion loss"""
    def __init__(self,
                 model_path='weights/raft-things.pth',
                 device='cuda',
                 small=False,
                 raft_orig=False):
        super().__init__()
        self.fix_raft = initialize_RAFT(
            model_path,
            device=device,
            small=small,
            raft_orig=raft_orig)

        for p in self.fix_raft.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()
        # print(gt_local_frames.shape)

        with torch.no_grad():
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            # print(gtlf_1.shape)

            _, gt_flows_forward = self.fix_raft(gtlf_1, gtlf_2, iters=iters)
            _, gt_flows_backward = self.fix_raft(gtlf_2, gtlf_1, iters=iters)

        gt_flows_forward = gt_flows_forward.view(b, l_t - 1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t - 1, 2, h, w)

        return gt_flows_forward, gt_flows_backward


def _test(raft_small: bool = False,
          raft_orig: bool = True):
    raft_iter = 20
    root_path = "../../../../pytorchcv_data/test"

    if raft_small:
        raft_model_file_name = "raft-small_.pth"
        y1_file_name = "y1_s.npy"
        y2_file_name = "y2_s.npy"
    else:
        raft_model_file_name = "raft-things_.pth"
        y1_file_name = "y1.npy"
        y2_file_name = "y2.npy"

    fix_raft = RAFT_bi(
        model_path=os.path.join(root_path, raft_model_file_name),
        device="cuda",
        small=raft_small,
        raft_orig=raft_orig)

    x_file_path = os.path.join(root_path, "x.npy")
    y1_file_path = os.path.join(root_path, y1_file_name)
    y2_file_path = os.path.join(root_path, y2_file_name)
    x = np.load(x_file_path)
    y1 = np.load(y1_file_path)
    y2 = np.load(y2_file_path)

    frames = torch.from_numpy(x).cuda()

    flows_f, flows_b = fix_raft(
        gt_local_frames=frames,
        iters=raft_iter)

    y1_ = flows_f.cpu().detach().numpy()
    y2_ = flows_b.cpu().detach().numpy()

    # np.save(os.path.join(root_path, "y1_s.npy"), np.ascontiguousarray(y1_))
    # np.save(os.path.join(root_path, "y2_s.npy"), np.ascontiguousarray(y2_))

    if not np.array_equal(y1, y1_):
        print("*")
    np.testing.assert_array_equal(y1, y1_)
    if not np.array_equal(y2, y2_):
        print("*")
    np.testing.assert_array_equal(y2, y2_)

    pass


if __name__ == "__main__":
    raft_orig = False
    _test(raft_small=True, raft_orig=raft_orig)
    _test(raft_small=False, raft_orig=raft_orig)
