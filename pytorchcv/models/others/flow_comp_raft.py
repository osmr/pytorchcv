import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from raft import RAFT


def initialize_RAFT(model_path='weights/raft-things.pth',
                    device='cuda',
                    small=False):
    """Initializes the RAFT model.
    """
    args = argparse.ArgumentParser()
    args.raft_model = model_path
    args.small = small
    args.mixed_precision = False
    args.alternate_corr = False
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model, map_location='cpu'))
    model = model.module

    model.to(device)

    return model


class RAFT_bi(nn.Module):
    """Flow completion loss"""
    def __init__(self,
                 model_path='weights/raft-things.pth',
                 device='cuda',
                 small=False):
        super().__init__()
        self.fix_raft = initialize_RAFT(
            model_path,
            device=device,
            small=small)

        for p in self.fix_raft.parameters():
            p.requires_grad = False

        self.l1_criterion = nn.L1Loss()
        self.eval()

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()
        # print(gt_local_frames.shape)

        with torch.no_grad():
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            # print(gtlf_1.shape)

            _, gt_flows_forward = self.fix_raft(gtlf_1, gtlf_2, iters=iters, test_mode=True)
            _, gt_flows_backward = self.fix_raft(gtlf_2, gtlf_1, iters=iters, test_mode=True)


        gt_flows_forward = gt_flows_forward.view(b, l_t-1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t-1, 2, h, w)

        return gt_flows_forward, gt_flows_backward


def _test():
    raft_small = False
    # raft_small = True

    raft_iter = 20
    root_path = "../../../../pytorchcv_data/test"

    if raft_small:
        raft_model_file_name = "raft-small.pth"
        y1_file_name = "y1_s.npy"
        y2_file_name = "y2_s.npy"
    else:
        raft_model_file_name = "raft-things.pth"
        y1_file_name = "y1.npy"
        y2_file_name = "y2.npy"

    fix_raft = RAFT_bi(
        model_path=os.path.join(root_path, raft_model_file_name),
        device="cuda",
        small=raft_small)

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
    _test()
