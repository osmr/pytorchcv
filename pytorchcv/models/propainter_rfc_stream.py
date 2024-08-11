"""
    Streaming mode of ProPainter (Recurrent Flow Completion), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['PPRFCDataLoader']

import torch
import torch.nn as nn
from typing import Sequence
from .raft_stream import (RAFTDataLoader, WindowBufferedDataLoader, WindowMultiIndex, calc_window_data_loader_index,
                          cat_window_data_loader_indices)
from .propainter_rfc import propainter_rfc, calc_bidirectional_opt_flow_completion_by_pprfc


class FlowCompWindowBufferedDataLoader(WindowBufferedDataLoader):
    """
    Optical flow completion window buffered data loader.

    Parameters
    ----------
    net: nn.Module
        Optical flow completion model.
    """
    def __init__(self,
                 net: nn.Module,
                 **kwargs):
        super(FlowCompWindowBufferedDataLoader, self).__init__(**kwargs)
        self.net = net

    def _load_data_items(self,
                         raw_data_chunk_list: tuple[Sequence, ...] | list[Sequence] | Sequence):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(sequence, ...) or list(sequence) or sequence
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Resulted data.
        """
        assert (len(raw_data_chunk_list) == 2)

        flows = raw_data_chunk_list[0]
        masks = raw_data_chunk_list[1]

        masks_forward = masks[:-1].contiguous()
        masks_backward = masks[1:].contiguous()
        flow_masks = torch.cat((masks_forward, masks_backward), dim=1)

        comp_flows, _ = calc_bidirectional_opt_flow_completion_by_pprfc(
            net=self.net,
            flows=flows,
            flow_masks=flow_masks)
        assert (len(comp_flows.shape) == 4)
        assert (comp_flows.shape[1] == 4)

        # torch.cuda.empty_cache()
        return comp_flows

    def _expand_buffer_by(self,
                          data_chunk: Sequence):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : sequence
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk], dim=0)


class PPRFCDataLoader(FlowCompWindowBufferedDataLoader):
    """
    Recurrent flow completion (from ProPainter) window buffered data loader.

    Parameters
    ----------
    flow_loader : Sequence
        Flow data loader (RAFT).
    mask_loader : sequence
        Mask data loader.
    pprfc_model_path : str or None, default None
        Path to ProPainter-RFC model parameters.
    """
    def __init__(self,
                 flow_loader: Sequence,
                 mask_loader: Sequence,
                 pprfc_model_path: str | None = None,
                 **kwargs):
        super(PPRFCDataLoader, self).__init__(
            data=[flow_loader, mask_loader],
            window_index=PPRFCDataLoader._calc_window_index(video_length=len(mask_loader)),
            net=PPRFCDataLoader._load_model(pprfc_model_path=pprfc_model_path),
            **kwargs)

    @staticmethod
    def _load_model(pprfc_model_path: str | None = None) -> nn.Module:
        """
        Load RAFT model.

        Parameters
        ----------
        pprfc_model_path : str or None, default None
            Path to ProPainter-RFC model parameters.

        Returns
        -------
        nn.Module
            ProPainter-RFC model.
        """
        if pprfc_model_path is not None:
            pprfc_net = propainter_rfc()
            pprfc_net.load_state_dict(torch.load(pprfc_model_path, map_location="cpu", weights_only=True))
        else:
            pprfc_net = propainter_rfc(pretrained=True)
        pprfc_net.eval()
        for p in pprfc_net.parameters():
            if p.requires_grad:
                pass
            p.requires_grad = False
        pprfc_net = pprfc_net.cuda()
        return pprfc_net

    @staticmethod
    def _calc_window_index(video_length: int) -> WindowMultiIndex:
        """
        Calculate window index.

        Parameters
        ----------
        video_length : int
            Number of frames.

        Returns
        -------
        WindowMultiIndex
            Desired window multiindex.
        """
        pprfc_flows_window_index = calc_window_data_loader_index(
            length=video_length - 1,
            window_size=80,
            padding=(5, 5),
            edge_mode="ignore")
        pprfc_mask_window_index = calc_window_data_loader_index(
            length=video_length,
            window_size=80,
            padding=(5, 6),
            edge_mode="ignore")
        pprfc_window_index = cat_window_data_loader_indices([pprfc_flows_window_index, pprfc_mask_window_index])
        return pprfc_window_index


def _test():
    pprfc_model_path = "../../../pytorchcv_data/test/propainter_rfc.pth"

    time = 140
    height = 240
    width = 432
    flows = torch.randn(time - 1, 4, height, width)
    masks = torch.randn(time, 2, height, width)

    flow_comp_loader = PPRFCDataLoader(
        flow_loader=flows,
        mask_loader=masks,
        pprfc_model_path=pprfc_model_path)

    video_length = time
    time_step = 10
    flow_loader_trim_pad = 3
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        comp_flows_i = flow_comp_loader[s:e]
        flow_comp_loader.trim_buffer_to(max(e - flow_loader_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
