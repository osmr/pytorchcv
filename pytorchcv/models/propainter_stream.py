"""
    Streaming mode of ProPainter, implemented in PyTorch.
    Original paper: 'ProPainter: Improving Propagation and Transformer for Video Inpainting,'
    https://arxiv.org/pdf/2309.03897.
"""

__all__ = ['ProPainterIterator']

import torch
import torch.nn as nn
from typing import Sequence
from .raft_stream import (WindowBufferedIterator, WindowMultiIndex, calc_sliding_window_iterator_index,
                          concat_window_iterator_indices)
from .propainter import propainter


class ImageTransIterator(WindowBufferedIterator):
    """
    Image transformer window buffered iterator.

    Parameters
    ----------
    net : nn.Module
        Video inpainting model.
    stride : int
        Stride value for sliding window.
    ref_stride : int
        Stride value for reference indices.
    num_refs : int
        Number of reference indices.
    """
    def __init__(self,
                 net: nn.Module,
                 stride,
                 ref_stride,
                 num_refs,
                 **kwargs):
        super(ImageTransIterator, self).__init__(**kwargs)
        self.net = net
        self.stride = stride
        self.ref_stride = ref_stride
        self.num_refs = num_refs

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

        updated_frames_masks = raw_data_chunk_list[0]
        masks = raw_data_chunk_list[1]
        comp_flows = raw_data_chunk_list[2]

        prop_frames, updated_masks = torch.split(updated_frames_masks, [3, 1], dim=1)

        win_pos = self.window_pos + 1
        s_idx = win_pos * self.stride

        neighbor_ids = ImageTransIterator._calc_image_trans_neighbor_index(
            mid_neighbor_id=s_idx,
            length=self.length,
            neighbor_stride=self.stride)
        ref_ids = ImageTransIterator._calc_image_trans_ref_index(
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

        # ref_neighbor_ids = sorted(neighbor_ids + ref_ids)
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

        # masked_frames0 = torch.from_numpy(np.load("../../../pytorchcv_data/testa/masked_frames0.npy")).cuda()
        # completed_flows0 = torch.from_numpy(np.load("../../../pytorchcv_data/testa/completed_flows0.npy")).cuda()
        # masks_in0 = torch.from_numpy(np.load("../../../pytorchcv_data/testa/masks_in0.npy")).cuda()
        # masks_updated0 = torch.from_numpy(np.load("../../../pytorchcv_data/testa/masks_updated0.npy")).cuda()
        # pred_img0a = torch.from_numpy(np.load("../../../pytorchcv_data/testa/pred_img0.npy")).cuda()
        # pred_frames0a = torch.from_numpy(np.load("../../../pytorchcv_data/testa/pred_frames0.npy")).cuda()
        # l_t0 = 6
        # pred_img0 = self.net(
        #     masked_frames=masked_frames0,
        #     completed_flows=completed_flows0,
        #     masks_in=masks_in0,
        #     masks_updated=masks_updated0,
        #     num_local_frames=l_t0)[0]
        # q1 = (pred_img0 - pred_img0a).abs().max()
        # q2 = (pred_img0 - pred_frames0a).abs().max()

        # torch.cuda.empty_cache()
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


class ProPainterIterator(ImageTransIterator):
    """
    ProPainter window buffered iterator.

    Parameters
    ----------
    image_prop_loader : Sequence
        Image propagation data loader (ProPainter-IP).
    mask_loader : sequence
        Mask data loader.
    flow_comp_loader : Sequence
        Flow completion data loader (ProPainter-RFC).
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
                 image_prop_loader: Sequence,
                 mask_loader: Sequence,
                 flow_comp_loader: Sequence,
                 pp_model_path: str | None = None,
                 pp_stride: int = 5,
                 pp_ref_stride: int = 10,
                 pp_ref_window_size: int = 80,
                 **kwargs):
        super(ProPainterIterator, self).__init__(
            data=[image_prop_loader, mask_loader, flow_comp_loader],
            window_index=ProPainterIterator._calc_window_index(
                video_length=len(mask_loader),
                pp_stride=pp_stride),
            net=ProPainterIterator._load_model(pp_model_path=pp_model_path),
            stride=pp_stride,
            ref_stride=pp_ref_stride,
            num_refs=(pp_ref_window_size // pp_ref_stride),
            **kwargs)

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


def _test():
    pp_model_path = "../../../pytorchcv_data/test/propainter.pth"

    time = 140
    height = 240
    width = 432
    updated_frames_masks = torch.randn(time, 5, height, width)
    masks = torch.randn(time, 2, height, width)
    comp_flows = torch.randn(time - 1, 4, height, width)

    image_trans_loader = ProPainterIterator(
        image_prop_loader=updated_frames_masks,
        mask_loader=masks,
        flow_comp_loader=comp_flows,
        pp_model_path=pp_model_path)

    video_length = time
    time_step = 10
    flow_loader_trim_pad = 3
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        pred_frames_i = image_trans_loader[s:e]
        assert (pred_frames_i is not None)
        image_trans_loader.trim_buffer_to(max(e - flow_loader_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
