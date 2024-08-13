"""
    Streaming mode of ProPainter, implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['ProPainterITIterator', 'ProPainterIterator']

import torch
import torch.nn as nn
from typing import Sequence
from .common.steam import (BufferedIterator, WindowBufferedIterator, WindowMultiIndex,
                           calc_sliding_window_iterator_index, concat_window_iterator_indices)
from .propainter import propainter


class ProPainterITIterator(WindowBufferedIterator):
    """
    Image transform (ProPainter-IT) window buffered iterator.

    Parameters
    ----------
    prop_framemasks : Sequence
        Image propagation iterator (ProPainter-IP).
    masks : sequence
        Mask iterator.
    comp_flows : Sequence
        Flow completion iterator (ProPainter-RFC).
    pp_model_path : str or None, default None
        Path to ProPainter model parameters.
    pp_stride : int, default 5
        Stride value for sliding window.
    pp_ref_stride : int, default 10
        Stride value for reference indices.
    pp_ref_window_size : int, default 80
        Window size for reference indices.
    """
    def __init__(self,
                 prop_framemasks: Sequence,
                 masks: Sequence,
                 comp_flows: Sequence,
                 pp_model_path: str | None = None,
                 pp_stride: int = 5,
                 pp_ref_stride: int = 10,
                 pp_ref_window_size: int = 80,
                 **kwargs):
        assert (len(masks) > 0)
        super(ProPainterITIterator, self).__init__(
            data=[prop_framemasks, masks, comp_flows],
            window_index=ProPainterITIterator._calc_window_index(
                video_length=len(masks),
                pp_stride=pp_stride),
            **kwargs)
        self.net = ProPainterITIterator._load_model(pp_model_path=pp_model_path)
        self.stride = pp_stride
        self.ref_stride = pp_ref_stride
        self.num_refs = pp_ref_window_size // pp_ref_stride

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

        prop_framemasks = raw_data_chunk_list[0]
        masks = raw_data_chunk_list[1]
        comp_flows = raw_data_chunk_list[2]

        prop_frames, updated_masks = torch.split(prop_framemasks, [3, 1], dim=1)

        win_pos = self.window_pos + 1
        s_idx = win_pos * self.stride

        neighbor_ids = ProPainterITIterator._calc_image_trans_neighbor_index(
            mid_neighbor_id=s_idx,
            length=self.length,
            neighbor_stride=self.stride)
        ref_ids = ProPainterITIterator._calc_image_trans_ref_index(
            mid_neighbor_id=s_idx,
            neighbor_ids=neighbor_ids,
            length=self.length,
            ref_stride=self.ref_stride,
            ref_num=self.num_refs)

        win_mmap = self.window_index[win_pos]

        assert (min(ref_ids) >= win_mmap.sources[0].start)
        assert (max(ref_ids) < win_mmap.sources[0].stop)
        assert (min(neighbor_ids) == win_mmap.sources[2].start)
        assert (max(neighbor_ids) == win_mmap.sources[2].stop)
        assert (len(neighbor_ids) == win_mmap.sources[2].stop - win_mmap.sources[2].start + 1)

        ref_neighbor_ids = neighbor_ids + ref_ids
        ref_neighbor_ids = [i - win_mmap.sources[0].start for i in ref_neighbor_ids]

        masked_frames = prop_frames[ref_neighbor_ids][None]
        masks_updated = updated_masks[ref_neighbor_ids][None]
        masks_in = masks[ref_neighbor_ids][None]
        completed_flows = comp_flows[None]

        l_t = len(comp_flows) + 1

        pred_frames = self.net(
            masked_frames=masked_frames,
            masks_updated=masks_updated,
            masks_in=masks_in,
            completed_flows=completed_flows,
            num_local_frames=l_t)

        pred_frames = pred_frames[0]

        return pred_frames

    def _calc_window_pose(self,
                          pos: int) -> int:
        """
        Calculate window pose.

        Parameters
        ----------
        pos : int
            Position of target data.

        Returns
        -------
        int
            Window position.
        """
        for win_pos in range(max(self.window_pos + 1, 0), self.window_length):
            win_start = self.window_index[win_pos].target.start
            if pos <= win_start:
                assert (win_pos > 0)
                return win_pos - 1
        return self.window_length - 1

    def _expand_buffer_by(self,
                          data_chunk: Sequence):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : sequence
            Data chunk.
        """
        win_pos = self.window_pos + 1
        win_mmap = self.window_index[win_pos]

        assert (win_mmap.target_start == 0)
        assert (win_mmap.target.start - self.start_pos >= 0)

        s = win_mmap.target.start - self.start_pos

        if not (s <= len(self.buffer)):
            pass

        assert (s <= len(self.buffer))

        if s == len(self.buffer):
            self.buffer = torch.cat([self.buffer, data_chunk], dim=0)
        else:
            buffer_tail = self.buffer[s:]
            buffer_tail_len = len(buffer_tail)
            assert (buffer_tail_len <= len(data_chunk))
            data_chunk1 = data_chunk[:buffer_tail_len]
            data_chunk2 = data_chunk[buffer_tail_len:]
            self.buffer[s:] = 0.5 * (buffer_tail + data_chunk1)
            self.buffer = torch.cat([self.buffer, data_chunk2], dim=0)

    @staticmethod
    def _calc_image_trans_neighbor_index(mid_neighbor_id,
                                         length,
                                         neighbor_stride):
        neighbor_ids = [i for i in range(
            max(0, mid_neighbor_id - neighbor_stride), min(length, mid_neighbor_id + neighbor_stride + 1))]
        return neighbor_ids

    @staticmethod
    def _calc_image_trans_ref_index(mid_neighbor_id,
                                    neighbor_ids,
                                    length,
                                    ref_stride,
                                    ref_num):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    @staticmethod
    def _load_model(pp_model_path: str | None = None) -> nn.Module:
        """
        Load ProPainter model.

        Parameters
        ----------
        pp_model_path : str or None, default None
            Path to ProPainter-RFC model parameters.

        Returns
        -------
        nn.Module
            ProPainter model.
        """
        if pp_model_path is not None:
            pp_net = propainter()
            pp_net.load_state_dict(torch.load(pp_model_path, map_location="cpu", weights_only=True))
        else:
            pp_net = propainter(pretrained=True)
        pp_net.eval()
        for p in pp_net.parameters():
            if p.requires_grad:
                pass
            p.requires_grad = False
        pp_net = pp_net.cuda()
        return pp_net

    @staticmethod
    def _calc_window_index(video_length: int,
                           pp_stride: int) -> WindowMultiIndex:
        """
        Calculate window index.

        Parameters
        ----------
        video_length : int
            Number of frames.
        pp_stride : int
            Stride value for sliding window.

        Returns
        -------
        WindowMultiIndex
            Desired window multiindex.
        """
        pp_ref_frames_window_index = calc_sliding_window_iterator_index(
            length=video_length,
            stride=pp_stride,
            src_padding=(40, 41),
            padding=(5, 6))
        pp_local_flows_window_index = calc_sliding_window_iterator_index(
            length=video_length,
            stride=pp_stride,
            src_padding=(5, 5),
            padding=(5, 6))
        pp_window_index = concat_window_iterator_indices([
            pp_ref_frames_window_index, pp_ref_frames_window_index, pp_local_flows_window_index])
        return pp_window_index


class ProPainterIterator(BufferedIterator):
    """
    Video inpainting (ProPainter) buffered iterator.

    Parameters
    ----------
    pred_frames : Sequence
        Image transform iterator (ProPainter-IT).
    frames : sequence
        Frame iterator.
    masks : sequence
        Mask iterator.
    """
    def __init__(self,
                 pred_frames: Sequence,
                 frames: Sequence,
                 masks: Sequence,
                 **kwargs):
        assert (len(frames) > 0)
        super(ProPainterIterator, self).__init__(
            data=[pred_frames, frames, masks],
            **kwargs)

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

        pred_frames = raw_data_chunk_list[0]
        frames = raw_data_chunk_list[1]
        masks = raw_data_chunk_list[2]

        vi_frames = pred_frames * masks + frames * (1 - masks)

        return vi_frames

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


def _test():
    pp_model_path = "../../../pytorchcv_data/test/propainter.pth"

    time = 140
    height = 240
    width = 432
    prop_framemasks = torch.randn(time, 5, height, width)
    masks = torch.randn(time, 1, height, width)
    comp_flows = torch.randn(time - 1, 4, height, width)
    frames = torch.randn(time, 3, height, width)

    pred_frame_iterator = ProPainterITIterator(
        prop_framemasks=prop_framemasks,
        masks=masks,
        comp_flows=comp_flows,
        pp_model_path=pp_model_path)

    video_length = time
    time_step = 10
    pred_frame_iterator_trim_pad = 6
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        pred_frames_i = pred_frame_iterator[s:e]
        assert (pred_frames_i is not None)
        pred_frame_iterator.trim_buffer_to(max(e - pred_frame_iterator_trim_pad, 0))
        torch.cuda.empty_cache()

    pred_frame_iterator.clear_buffer()

    vi_frame_iterator = ProPainterIterator(
        pred_frames=pred_frame_iterator,
        frames=frames,
        masks=masks)

    vi_frame_iterator_trim_pad = 2
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        vi_frames_i = vi_frame_iterator[s:e]
        assert (vi_frames_i is not None)
        vi_frame_iterator.trim_buffer_to(max(e - vi_frame_iterator_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
