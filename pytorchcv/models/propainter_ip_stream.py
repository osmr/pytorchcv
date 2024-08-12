"""
    Streaming mode of ProPainter (Image Propagation), implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['PPIPIterator']

import torch
import torch.nn as nn
from typing import Sequence
from .raft_stream import (WindowBufferedIterator, WindowMultiIndex, calc_serial_window_iterator_index,
                          concat_window_iterator_indices)
from .propainter_ip import propainter_ip


class ImagePropIterator(WindowBufferedIterator):
    """
    Image propagation window buffered iterator.

    Parameters
    ----------
    net : nn.Module
        Image propagation model.
    """
    def __init__(self,
                 net: nn.Module,
                 **kwargs):
        super(ImagePropIterator, self).__init__(**kwargs)
        self.net = net

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


class PPIPIterator(ImagePropIterator):
    """
    Image propagation (from ProPainter) window buffered iterator.

    Parameters
    ----------
    frames : sequence
        Frame iterator.
    masks : sequence
        Mask iterator.
    comp_flows : Sequence
        Flow completion iterator (ProPainter-RFC).
    """
    def __init__(self,
                 frames: Sequence,
                 masks: Sequence,
                 comp_flows: Sequence,
                 **kwargs):
        super(PPIPIterator, self).__init__(
            data=[frames, masks, comp_flows],
            window_index=PPIPIterator._calc_window_index(video_length=len(masks)),
            net=PPIPIterator._load_model(),
            **kwargs)

    @staticmethod
    def _load_model() -> nn.Module:
        """
        Load ProPainter-IP model.

        Returns
        -------
        nn.Module
            ProPainter-IP model.
        """
        ppip_net = propainter_ip()
        ppip_net.eval()
        ppip_net = ppip_net.cuda()
        return ppip_net

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
        ppip_images_window_index = calc_serial_window_iterator_index(
            length=video_length,
            window_size=80,
            padding=(10, 10),
            edge_mode="ignore")
        ppip_flows_window_index = calc_serial_window_iterator_index(
            length=video_length - 1,
            window_size=80,
            padding=(10, 9),
            edge_mode="ignore")
        ppip_window_index = concat_window_iterator_indices([
            ppip_images_window_index, ppip_images_window_index, ppip_flows_window_index])
        return ppip_window_index


def _test():
    time = 140
    height = 240
    width = 432
    frames = torch.randn(time, 3, height, width)
    masks = torch.randn(time, 2, height, width)
    comp_flows = torch.randn(time - 1, 4, height, width)

    image_prop_loader = PPIPIterator(
        frames=frames,
        masks=masks,
        comp_flows=comp_flows)

    video_length = time
    time_step = 10
    flow_loader_trim_pad = 3
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        updated_frames_masks_i = image_prop_loader[s:e]
        image_prop_loader.trim_buffer_to(max(e - flow_loader_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
