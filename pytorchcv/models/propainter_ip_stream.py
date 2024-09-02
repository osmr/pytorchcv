"""
    Streaming mode of ProPainter (Image Propagation), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['ProPainterIPSequencer']

import torch
import torch.nn as nn
from typing import Sequence
from pytorchcv.models.common.steam import (WindowBufferedSequencer, WindowMultiIndex, calc_serial_window_sequencer_index,
                                           concat_window_sequencer_indices)
from pytorchcv.models.propainter_ip import propainter_ip


class ProPainterIPSequencer(WindowBufferedSequencer):
    """
    Image propagation (ProPainter-IP) window buffered sequencer.

    Parameters
    ----------
    frames : sequence
        Frame sequencer.
    masks : sequence
        Mask sequencer.
    comp_flows : Sequence
        Flow completion sequencer (ProPainter-RFC).
    use_cuda : bool, default True
        Whether to use CUDA.
    window_size : int, default 80
        Window size.
    padding : int, default 10
        Padding value.
    """
    def __init__(self,
                 frames: Sequence,
                 masks: Sequence,
                 comp_flows: Sequence,
                 use_cuda: bool = True,
                 window_size: int = 80,
                 padding: int = 10,
                 **kwargs):
        assert (len(frames) > 0)
        super(ProPainterIPSequencer, self).__init__(
            data=[frames, masks, comp_flows],
            window_index=ProPainterIPSequencer._calc_window_index(
                video_length=len(masks),
                window_size=window_size,
                padding=padding),
            **kwargs)
        self.net = ProPainterIPSequencer._load_model(use_cuda=use_cuda)

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
        assert (len(raw_data_chunk_list) == 3)

        frames = raw_data_chunk_list[0]
        masks = raw_data_chunk_list[1]
        comp_flows = raw_data_chunk_list[2]

        prop_frames, updated_masks = self.net(
            frames=frames,
            masks=masks,
            comp_flows=comp_flows,
            interpolation="nearest")

        assert (len(prop_frames.shape) == 4)
        assert (len(updated_masks.shape) == 4)

        updated_frames_masks = torch.cat((prop_frames, updated_masks), dim=1)
        assert (updated_frames_masks.shape[1] == 4)

        return updated_frames_masks

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
    def _load_model(use_cuda: bool = True) -> nn.Module:
        """
        Load ProPainter-IP model.

        Parameters
        ----------
        use_cuda : bool, default True
            Whether to use CUDA.

        Returns
        -------
        nn.Module
            ProPainter-IP model.
        """
        ppip_net = propainter_ip()
        ppip_net.eval()
        if use_cuda:
            ppip_net = ppip_net.cuda()
        return ppip_net

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
        ppip_images_window_index = calc_serial_window_sequencer_index(
            length=video_length,
            target_length=video_length,
            window_size=window_size,
            padding=(padding, padding),
            edge_mode="ignore")
        ppip_flows_window_index = calc_serial_window_sequencer_index(
            length=video_length - 1,
            target_length=video_length,
            window_size=window_size,
            padding=(padding, padding - 1),
            edge_mode="ignore")
        ppip_window_index = concat_window_sequencer_indices([
            ppip_images_window_index, ppip_images_window_index, ppip_flows_window_index])
        return ppip_window_index


def _test():
    time = 140
    height = 240
    width = 432
    frames = torch.randn(time, 3, height, width)
    masks = torch.randn(time, 1, height, width)
    comp_flows = torch.randn(time - 1, 4, height, width)

    prop_framemask_sequencer = ProPainterIPSequencer(
        frames=frames,
        masks=masks,
        comp_flows=comp_flows,
        use_cuda=True)

    video_length = time
    time_step = 10
    prop_framemask_sequencer_trim_pad = 2
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        prop_framemasks_i = prop_framemask_sequencer[s:e]
        assert (prop_framemasks_i is not None)
        prop_framemask_sequencer.trim_buffer_to(max(e - prop_framemask_sequencer_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
