import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor0 import BasicEncoder, SmallEncoder
from corr import CorrBlock


def coords_grid(batch,
                ht,
                wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow,
            mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class RAFT0(nn.Module):
    def __init__(self,
                 small: bool,
                 dropout: float = 0.0):
        super(RAFT0, self).__init__()
        self.small = small
        self.dropout = dropout

        if self.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_levels = 4
            self.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4

        # feature network, context network, and update block
        if self.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn="instance", dropout=self.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn="none", dropout=self.dropout)
            self.update_block = SmallUpdateBlock(
                corr_levels=self.corr_levels,
                corr_radius=self.corr_radius,
                hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=self.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn="batch", dropout=self.dropout)
            self.update_block = BasicUpdateBlock(
                corr_levels=self.corr_levels,
                corr_radius=self.corr_radius,
                hidden_dim=hdim)

    @staticmethod
    def initialize_flow(img):
        """
        Flow is represented as difference between two coordinate grids flow = coords1 - coords0.
        """
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    @staticmethod
    def upsample_flow(flow, mask):
        """
        Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination.
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None):
        """
        Estimate optical flow between pair of frames.
        """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

        return coords1 - coords0, flow_up