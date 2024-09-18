"""
    Streaming mode of ProPainter (Recurrent Flow Completion), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['ProPainterRFCSequencer']

import torch
import torch.nn as nn
from typing import Sequence
from pytorchcv.models.common.stream import (WindowBufferedSequencer, WindowMultiIndex, calc_serial_window_sequencer_index,
                                            concat_window_sequencer_indices)
from pytorchcv.models.propainter_rfc import propainter_rfc, calc_bidirectional_opt_flow_completion_by_pprfc


class ProPainterRFCSequencer(WindowBufferedSequencer):
    """
    Optical flow completion (ProPainter-RFC) window buffered sequencer.

    Parameters
    ----------
    flows : Sequence
        Flow sequencer (RAFT).
    masks : sequence
        Mask sequencer.
    pprfc_model : nn.Module or str or None, default None
        ProPainter-RFC model or path to ProPainter-RFC model parameters.
    use_cuda : bool, default True
        Whether to use CUDA.
    window_size : int, default 80
        Window size.
    padding : int, default 5
        Padding value.
    """
    def __init__(self,
                 flows: Sequence,
                 masks: Sequence,
                 pprfc_model: nn.Module | str | None = None,
                 use_cuda: bool = True,
                 window_size: int = 80,
                 padding: int = 5,
                 **kwargs):
        assert (len(masks) > 0)
        super(ProPainterRFCSequencer, self).__init__(
            data=[flows, masks],
            window_index=ProPainterRFCSequencer._calc_window_index(
                video_length=len(masks),
                window_size=window_size,
                padding=padding),
            **kwargs)
        self.net = ProPainterRFCSequencer._load_model(
            pprfc_model=pprfc_model,
            use_cuda=use_cuda)

    def _calc_data_items(self,
                         raw_data_chunk_list: tuple[Sequence, ...] | list[Sequence] | Sequence) -> Sequence:
        """
        Calculate/load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(sequence, ...) or list(sequence) or sequence
            List of source data chunks.

        Returns
        -------
        sequence
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

    @staticmethod
    def _load_model(pprfc_model: nn.Module | str | None = None,
                    use_cuda: bool = True) -> nn.Module:
        """
        Load ProPainter-RFC model.

        Parameters
        ----------
        pprfc_model : nn.Module or str or None, default None
            ProPainter-RFC model or path to ProPainter-RFC model parameters.
        use_cuda : bool, default True
            Whether to use CUDA.

        Returns
        -------
        nn.Module
            ProPainter-RFC model.
        """
        if isinstance(pprfc_model, nn.Module):
            return pprfc_model
        if pprfc_model is not None:
            pprfc_net = propainter_rfc()
            pprfc_net.load_state_dict(torch.load(pprfc_model, map_location="cpu", weights_only=True))
        else:
            pprfc_net = propainter_rfc(pretrained=True)
        pprfc_net.eval()
        for p in pprfc_net.parameters():
            if p.requires_grad:
                pass
            p.requires_grad = False
        if use_cuda:
            pprfc_net = pprfc_net.cuda()
        return pprfc_net

    @staticmethod
    def _calc_window_index(video_length: int,
                           window_size: int,
                           padding: int) -> WindowMultiIndex:
        """
        Calculate window index.

        Parameters
        ----------
        video_length : int
            Number of frames.
        window_size : int
            Window size.
        padding : int
            Padding value.

        Returns
        -------
        WindowMultiIndex
            Desired window multiindex.
        """
        assert (window_size > 0)
        pprfc_flows_window_index = calc_serial_window_sequencer_index(
            length=video_length - 1,
            target_length=video_length,
            window_size=window_size,
            padding=(padding, padding),
            edge_mode="ignore")
        pprfc_mask_window_index = calc_serial_window_sequencer_index(
            length=video_length,
            target_length=video_length,
            window_size=window_size,
            padding=(padding, padding + 1),
            edge_mode="ignore")
        pprfc_window_index = concat_window_sequencer_indices([pprfc_flows_window_index, pprfc_mask_window_index])
        return pprfc_window_index


def _test():
    time = 140
    height = 240
    width = 432
    flows = torch.randn(time - 1, 4, height, width).cuda()
    masks = torch.randn(time, 1, height, width).cuda()

    comp_flow_sequencer = ProPainterRFCSequencer(
        flows=flows,
        masks=masks,
        pprfc_model=None,
        use_cuda=True)

    video_length = time
    time_step = 10
    comp_flow_sequencer_trim_pad = 2
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        comp_flows_i = comp_flow_sequencer[s:e]
        assert (comp_flows_i is not None)
        comp_flow_sequencer.trim_buffer_to(max(e - comp_flow_sequencer_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
