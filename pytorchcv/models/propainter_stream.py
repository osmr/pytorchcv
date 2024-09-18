"""
    Streaming mode of ProPainter, implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['ProPainterITSequencer', 'ProPainterIMSequencer', 'ProPainterIterator']

import torch
import torch.nn as nn
from typing import Sequence
from pytorchcv.models.common.stream import (Sequencer, BufferedSequencer, WindowBufferedSequencer, WindowMultiIndex,
                                            calc_sliding_window_sequencer_index, concat_window_sequencer_indices)
from pytorchcv.models.propainter import propainter
from pytorchcv.models.raft_stream import RAFTSequencer
from pytorchcv.models.propainter_rfc_stream import ProPainterRFCSequencer
from pytorchcv.models.propainter_ip_stream import ProPainterIPSequencer


class ProPainterITSequencer(WindowBufferedSequencer):
    """
    Image transform (ProPainter-IT) window buffered sequencer.

    Parameters
    ----------
    prop_framemasks : Sequence
        Image propagation sequencer (ProPainter-IP).
    masks : sequence
        Mask sequencer.
    comp_flows : Sequence
        Flow completion sequencer (ProPainter-RFC).
    pp_model : nn.Module or str or None, default None
        ProPainter model or path to ProPainter model parameters.
    use_cuda : bool, default True
        Whether to use CUDA.
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
                 use_cuda: bool = True,
                 pp_stride: int = 5,
                 pp_ref_stride: int = 10,
                 pp_ref_window_size: int = 80,
                 **kwargs):
        assert (len(masks) > 0)
        super(ProPainterITSequencer, self).__init__(
            data=[prop_framemasks, masks, comp_flows],
            window_index=ProPainterITSequencer._calc_window_index(
                video_length=len(masks),
                pp_stride=pp_stride,
                pp_ref_window_size=pp_ref_window_size),
            **kwargs)
        self.net = ProPainterITSequencer._load_model(
            pp_model=pp_model,
            use_cuda=use_cuda)
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

        neighbor_ids = ProPainterITSequencer._calc_image_trans_neighbor_index(
            mid_neighbor_id=s_idx,
            length=self.length,
            neighbor_stride=self.stride)
        ref_ids = ProPainterITSequencer._calc_image_trans_ref_index(
            mid_neighbor_id=s_idx,
            neighbor_ids=neighbor_ids,
            length=self.length,
            ref_stride=self.ref_stride,
            ref_num=self.num_refs)

        win_mmap = self.window_index[win_pos]

        assert ((not ref_ids) or (min(ref_ids) >= win_mmap.sources[0].start))
        assert ((not ref_ids) or (max(ref_ids) < win_mmap.sources[0].stop))
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
    def _load_model(pp_model: nn.Module | str | None = None,
                    use_cuda: bool = True) -> nn.Module:
        """
        Load ProPainter model.

        Parameters
        ----------
        pp_model : nn.Module or str or None, default None
            ProPainter model or path to ProPainter model parameters.
        use_cuda : bool, default True
            Whether to use CUDA.

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
        if use_cuda:
            pp_net = pp_net.cuda()
        return pp_net

    @staticmethod
    def _calc_window_index(video_length: int,
                           pp_stride: int,
                           pp_ref_window_size: int) -> WindowMultiIndex:
        """
        Calculate window index.

        Parameters
        ----------
        video_length : int
            Number of frames.
        pp_stride : int
            Stride value for sliding window.
        pp_ref_window_size : int
            Window size for reference indices.

        Returns
        -------
        WindowMultiIndex
            Desired window multiindex.
        """
        assert (pp_ref_window_size % 2 == 0)
        pp_ref_frames_window_index = calc_sliding_window_sequencer_index(
            length=video_length,
            stride=pp_stride,
            src_padding=(pp_ref_window_size // 2, pp_ref_window_size // 2 + 1),
            padding=(pp_stride, pp_stride + 1))
        pp_local_flows_window_index = calc_sliding_window_sequencer_index(
            length=video_length,
            stride=pp_stride,
            src_padding=(pp_stride, pp_stride),
            padding=(pp_stride, pp_stride + 1))
        pp_window_index = concat_window_sequencer_indices([
            pp_ref_frames_window_index, pp_ref_frames_window_index, pp_local_flows_window_index])
        return pp_window_index


class ProPainterIMSequencer(Sequencer):
    """
    Inpaint Masking (ProPainter-IM) sequencer.

    Parameters
    ----------
    trans_frames : Sequence
        Image transform sequencer (ProPainter-IT).
    frames : sequence
        Frame sequencer.
    masks : sequence
        Mask sequencer.
    """
    def __init__(self,
                 trans_frames: Sequence,
                 frames: Sequence,
                 masks: Sequence):
        assert (len(frames) > 0)
        super(ProPainterIMSequencer, self).__init__(data=[trans_frames, frames, masks])

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


class ProPainterIterator:
    """
    Video Inpainting (ProPainter) iterator.

    Parameters
    ----------
    frames : BufferedSequencer
        Frame sequencer.
    masks : BufferedSequencer
        Mask sequencer.
    raft_model : nn.Module or str or None, default None
        RAFT model or path to RAFT model parameters.
    pprfc_model : nn.Module or str or None, default None
        ProPainter-RFC model or path to ProPainter-RFC model parameters.
    pp_model : nn.Module or str or None, default None
        ProPainter model or path to ProPainter model parameters.
    use_cuda : bool, default True
        Whether to use CUDA.
    raft_window_size : int or None, default None
        RAFT sequencer window size.
    pp_window_size : int, default 80
        ProPainter's blocks window size.
    pp_stride : int, default 5
        ProPainter's stride value for sliding window.
    step : int, default 10
        Main iteration window size.
    """
    def __init__(self,
                 frames: BufferedSequencer,
                 masks: BufferedSequencer,
                 raft_model: nn.Module | str | None = None,
                 pprfc_model: nn.Module | str | None = None,
                 pp_model: nn.Module | str | None = None,
                 use_cuda: bool = True,
                 raft_window_size: int | None = None,
                 pp_window_size: int = 80,
                 pp_stride: int = 5,
                 step: int = 10):
        super(ProPainterIterator, self).__init__()
        assert (len(frames) > 0)
        assert (len(frames) == len(masks))
        assert (step > 0)
        assert isinstance(frames, BufferedSequencer)
        assert isinstance(masks, BufferedSequencer)

        self.video_length = len(frames)
        self.step = step

        self.frames = frames
        self.masks = masks

        self.flow_sequencer = RAFTSequencer(
            frames=frames,
            raft_model=raft_model,
            use_cuda=use_cuda,
            window_size=raft_window_size)
        self.comp_flow_sequencer = ProPainterRFCSequencer(
            flows=self.flow_sequencer,
            masks=masks,
            pprfc_model=pprfc_model,
            use_cuda=use_cuda,
            window_size=pp_window_size)
        self.prop_framemask_sequencer = ProPainterIPSequencer(
            frames=frames,
            masks=masks,
            comp_flows=self.comp_flow_sequencer,
            use_cuda=use_cuda,
            window_size=pp_window_size)
        self.trans_frame_sequencer = ProPainterITSequencer(
            prop_framemasks=self.prop_framemask_sequencer,
            masks=masks,
            comp_flows=self.comp_flow_sequencer,
            pp_model=pp_model,
            use_cuda=use_cuda,
            pp_ref_window_size=pp_window_size)
        self.inp_frame_sequencer = ProPainterIMSequencer(
            trans_frames=self.trans_frame_sequencer,
            frames=frames,
            masks=masks)
        self.main_sequencer = self.inp_frame_sequencer

        self.trans_frame_sequencer_trim_pad = 2
        self.prop_framemask_sequencer_trim_pad = pp_window_size // 2 - pp_stride
        self.comp_flow_sequencer_trim_pad = 2
        self.flow_sequencer_trim_pad = 2
        self.mask_sequencer_trim_pad = pp_window_size // 2 - pp_stride
        self.frame_sequencer_trim_pad = 2

    def __iter__(self):
        self.s = -self.step

        self.trans_frame_sequencer.clear_buffer()
        self.prop_framemask_sequencer.clear_buffer()
        self.comp_flow_sequencer.clear_buffer()
        self.flow_sequencer.clear_buffer()
        self.masks.clear_buffer()
        self.frames.clear_buffer()

        torch.cuda.empty_cache()

        return self

    def __next__(self):
        if self.s is None:
            raise StopIteration

        self.s = min(self.s + self.step, self.video_length - 1)
        e = min(self.s + self.step, self.video_length)

        data = self.main_sequencer[self.s:e]

        self.trans_frame_sequencer.trim_buffer_to(max(e - self.trans_frame_sequencer_trim_pad, 0))
        self.prop_framemask_sequencer.trim_buffer_to(max(e - self.prop_framemask_sequencer_trim_pad, 0))
        self.comp_flow_sequencer.trim_buffer_to(max(e - self.comp_flow_sequencer_trim_pad, 0))
        self.flow_sequencer.trim_buffer_to(max(e - self.flow_sequencer_trim_pad, 0))
        self.masks.trim_buffer_to(max(e - self.mask_sequencer_trim_pad, 0))
        self.frames.trim_buffer_to(max(e - self.frame_sequencer_trim_pad, 0))

        torch.cuda.empty_cache()

        if e == self.video_length:
            self.s = None

        return data


class TensorSequencer(BufferedSequencer):
    """
    Tensor buffered sequencer.

    Parameters
    ----------
    data : torch.Tensor
        Source data or data iterators (arguments of calculator).
    """
    def __init__(self,
                 data: torch.Tensor):
        assert isinstance(data, torch.Tensor)
        super(TensorSequencer, self).__init__(data=data)

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
        assert (len(raw_data_chunk_list) == 1)
        return raw_data_chunk_list[0]

    def _expand_buffer_by(self,
                          data_chunk: Sequence):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : sequence
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk])


def _test():
    time = 140
    height = 240
    width = 432
    prop_framemasks = torch.randn(time, 4, height, width).cuda()
    masks = torch.randn(time, 1, height, width).cuda()
    comp_flows = torch.randn(time - 1, 4, height, width).cuda()
    frames = torch.randn(time, 3, height, width).cuda()

    trans_frame_sequencer = ProPainterITSequencer(
        prop_framemasks=prop_framemasks,
        masks=masks,
        comp_flows=comp_flows,
        pp_model=None,
        use_cuda=True)

    video_length = time
    time_step = 10
    trans_frame_sequencer_trim_pad = 2
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        trans_frames_i = trans_frame_sequencer[s:e]
        assert (trans_frames_i is not None)
        trans_frame_sequencer.trim_buffer_to(max(e - trans_frame_sequencer_trim_pad, 0))
        torch.cuda.empty_cache()

    trans_frame_sequencer.clear_buffer()

    inp_frame_sequencer = ProPainterIMSequencer(
        trans_frames=trans_frame_sequencer,
        frames=frames,
        masks=masks)

    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        inp_frames_i = inp_frame_sequencer[s:e]
        assert (inp_frames_i is not None)

    vi_sequencer = ProPainterIterator(
        frames=TensorSequencer(data=frames),
        masks=TensorSequencer(data=masks),
        raft_model=None,
        pprfc_model=None,
        pp_model=None,
        use_cuda=True)

    for frames_i in vi_sequencer:
        assert (frames_i is not None)

    pass


if __name__ == "__main__":
    _test()
