"""
    Streaming mode of ProPainter, implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['ProPainterITIterator', 'ProPainterIMIterator', 'ProPainterIterator']

import torch
import torch.nn as nn
from typing import Sequence
from .common.steam import (BufferedIterator, WindowBufferedIterator, WindowMultiIndex,
                           calc_sliding_window_iterator_index, concat_window_iterator_indices)
from .propainter import propainter
from .raft_stream import RAFTIterator
from .propainter_rfc_stream import ProPainterRFCIterator
from .propainter_ip_stream import ProPainterIPIterator


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
    pp_model : nn.Module or str or None, default None
        ProPainter model or path to ProPainter model parameters.
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
                 pp_model: nn.Module | str | None = None,
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
        self.net = ProPainterITIterator._load_model(pp_model=pp_model)
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

        trans_frames = self.net(
            masked_frames=masked_frames,
            masks_updated=masks_updated,
            masks_in=masks_in,
            completed_flows=completed_flows,
            num_local_frames=l_t)

        trans_frames = trans_frames[0]

        return trans_frames

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
    def _load_model(pp_model: nn.Module | str | None = None) -> nn.Module:
        """
        Load ProPainter model.

        Parameters
        ----------
        pp_model : nn.Module or str or None, default None
            ProPainter model or path to ProPainter model parameters.

        Returns
        -------
        nn.Module
            ProPainter model.
        """
        if isinstance(pp_model, nn.Module):
            return pp_model
        if pp_model is not None:
            pp_net = propainter()
            pp_net.load_state_dict(torch.load(pp_model, map_location="cpu", weights_only=True))
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


class ProPainterIMIterator(BufferedIterator):
    """
    Inpaint Masking (ProPainter-IM) buffered iterator.

    Parameters
    ----------
    trans_frames : Sequence
        Image transform iterator (ProPainter-IT).
    frames : sequence
        Frame iterator.
    masks : sequence
        Mask iterator.
    """
    def __init__(self,
                 trans_frames: Sequence,
                 frames: Sequence,
                 masks: Sequence,
                 **kwargs):
        assert (len(frames) > 0)
        super(ProPainterIMIterator, self).__init__(
            data=[trans_frames, frames, masks],
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

        trans_frames = raw_data_chunk_list[0]
        frames = raw_data_chunk_list[1]
        masks = raw_data_chunk_list[2]

        inp_frames = trans_frames * masks + frames * (1 - masks)

        return inp_frames

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


class ProPainterIterator:
    """
    Video Inpainting (ProPainter) iterator.

    Parameters
    ----------
    frames : BufferedIterator
        Frame iterator.
    masks : BufferedIterator
        Mask iterator.
    raft_model : nn.Module or str or None, default None
        RAFT model or path to RAFT model parameters.
    pprfc_model : nn.Module or str or None, default None
        ProPainter-RFC model or path to ProPainter-RFC model parameters.
    pp_model : nn.Module or str or None, default None
        ProPainter model or path to ProPainter model parameters.
    step : int, default 10
        Iteration window size.
    """
    def __init__(self,
                 frames: BufferedIterator,
                 masks: BufferedIterator,
                 raft_model: nn.Module | str | None = None,
                 pprfc_model: nn.Module | str | None = None,
                 pp_model: nn.Module | str | None = None,
                 step: int = 10):
        super(ProPainterIterator, self).__init__()
        assert (len(frames) > 0)
        assert (len(frames) == len(masks))
        assert (step > 0)

        self.video_length = len(frames)
        self.step = step

        self.frames = frames
        self.masks = masks

        self.flow_iterator = RAFTIterator(
            frames=frames,
            raft_model=raft_model)
        self.comp_flow_iterator = ProPainterRFCIterator(
            flows=self.flow_iterator,
            masks=masks,
            pprfc_model=pprfc_model)
        self.prop_framemask_iterator = ProPainterIPIterator(
            frames=frames,
            masks=masks,
            comp_flows=self.comp_flow_iterator)
        self.trans_frame_iterator = ProPainterITIterator(
            prop_framemasks=self.prop_framemask_iterator,
            masks=masks,
            comp_flows=self.comp_flow_iterator,
            pp_model=pp_model)
        self.inp_frame_iterator = ProPainterIMIterator(
            trans_frames=self.trans_frame_iterator,
            frames=frames,
            masks=masks)

        self.inp_frame_iterator_trim_pad = 2
        self.trans_frame_iterator_trim_pad = 6
        self.prop_framemask_iterator_trim_pad = 35
        self.comp_flow_iterator_trim_pad = 3
        self.flow_iterator_trim_pad = 3
        self.mask_iterator_trim_pad = 35
        self.frame_iterator_trim_pad = 2

    def __iter__(self):
        self.s = -self.step

        self.inp_frame_iterator.clear_buffer()
        self.trans_frame_iterator.clear_buffer()
        self.prop_framemask_iterator.clear_buffer()
        self.comp_flow_iterator.clear_buffer()
        self.flow_iterator.clear_buffer()
        self.masks.clear_buffer()
        self.frames.clear_buffer()

        torch.cuda.empty_cache()

        return self

    def __next__(self):
        if self.s is None:
            raise StopIteration

        self.s = min(self.s + self.step, self.video_length - 1)
        e = min(self.s + self.step, self.video_length)

        data = self.inp_frame_iterator[self.s:e]

        self.inp_frame_iterator.trim_buffer_to(max(e - self.inp_frame_iterator_trim_pad, 0))
        self.trans_frame_iterator.trim_buffer_to(max(e - self.trans_frame_iterator_trim_pad, 0))
        self.prop_framemask_iterator.trim_buffer_to(max(e - self.prop_framemask_iterator_trim_pad, 0))
        self.comp_flow_iterator.trim_buffer_to(max(e - self.comp_flow_iterator_trim_pad, 0))
        self.flow_iterator.trim_buffer_to(max(e - self.flow_iterator_trim_pad, 0))
        self.masks.trim_buffer_to(max(e - self.mask_iterator_trim_pad, 0))
        self.frames.trim_buffer_to(max(e - self.frame_iterator_trim_pad, 0))

        torch.cuda.empty_cache()

        if e == self.video_length:
            self.s = None

        return data


def _test():
    time = 140
    height = 240
    width = 432
    prop_framemasks = torch.randn(time, 5, height, width)
    masks = torch.randn(time, 1, height, width)
    comp_flows = torch.randn(time - 1, 4, height, width)
    frames = torch.randn(time, 3, height, width)

    trans_frame_iterator = ProPainterITIterator(
        prop_framemasks=prop_framemasks,
        masks=masks,
        comp_flows=comp_flows,
        pp_model=None)

    video_length = time
    time_step = 10
    trans_frame_iterator_trim_pad = 6
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        trans_frames_i = trans_frame_iterator[s:e]
        assert (trans_frames_i is not None)
        trans_frame_iterator.trim_buffer_to(max(e - trans_frame_iterator_trim_pad, 0))
        torch.cuda.empty_cache()

    trans_frame_iterator.clear_buffer()

    inp_frame_iterator = ProPainterIMIterator(
        trans_frames=trans_frame_iterator,
        frames=frames,
        masks=masks)

    inp_frame_iterator_trim_pad = 2
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        inp_frames_i = inp_frame_iterator[s:e]
        assert (inp_frames_i is not None)
        inp_frame_iterator.trim_buffer_to(max(e - inp_frame_iterator_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
