"""
    Streaming mode of RAFT, implemented in PyTorch.
    Original paper: 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,'
    https://arxiv.org/pdf/2003.12039.
"""

__all__ = ['RAFTIterator']

import torch
import torch.nn as nn
from typing import Sequence
from .common.steam import WindowBufferedIterator, WindowIndex, calc_serial_window_iterator_index
from .raft import raft_things, calc_bidirectional_optical_flow_on_video_by_raft


class FlowIterator(WindowBufferedIterator):
    """
    Optical flow window buffered iterator.

    Parameters
    ----------
    net : nn.Module
        Optical flow model.
    """
    def __init__(self,
                 net: nn.Module,
                 **kwargs):
        super(FlowIterator, self).__init__(**kwargs)
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
        assert (len(raw_data_chunk_list) == 1)

        frames = raw_data_chunk_list[0]
        flows = calc_bidirectional_optical_flow_on_video_by_raft(
            net=self.net,
            frames=frames)
        assert (len(flows.shape) == 4)
        assert (flows.shape[1] == 4)

        return flows

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


class RAFTIterator(FlowIterator):
    """
    RAFT window buffered iterator.

    Parameters
    ----------
    frames : sequence
        Frame iterator.
    raft_model_path : str or None, default None
        Path to RAFT model parameters.
    raft_iters : int, default 20
        Number of iterations in RAFT.
    """
    def __init__(self,
                 frames: Sequence,
                 raft_model_path: str | None = None,
                 raft_iters: int = 20,
                 **kwargs):
        assert (len(frames) > 0)
        super(RAFTIterator, self).__init__(
            data=frames,
            window_index=RAFTIterator._calc_window_index(
                video_length=len(frames),
                frame_size=frames[0].shape[1:]),
            net=RAFTIterator._load_model(
                raft_model_path=raft_model_path,
                raft_iters=raft_iters),
            **kwargs)

    @staticmethod
    def _load_model(raft_model_path: str | None = None,
                    raft_iters: int = 20) -> nn.Module:
        """
        Load RAFT model.

        Parameters
        ----------
        raft_model_path : str or None, default None
            Path to RAFT model parameters.
        raft_iters : int, default 20
            Number of iterations in RAFT.

        Returns
        -------
        nn.Module
            RAFT model.
        """
        if raft_model_path is not None:
            raft_net = raft_things(
                in_normalize=False,
                iters=raft_iters)
            raft_net.load_state_dict(torch.load(raft_model_path, map_location="cpu", weights_only=True))
        else:
            raft_net = raft_things(
                pretrained=True,
                in_normalize=False,
                iters=raft_iters)
        raft_net.eval()
        for p in raft_net.parameters():
            if p.requires_grad:
                pass
            p.requires_grad = False
        raft_net = raft_net.cuda()
        return raft_net

    @staticmethod
    def _calc_window_index(video_length: int,
                           frame_size: tuple[int, int]) -> WindowIndex:
        """
        Calculate window index.

        Parameters
        ----------
        video_length : int
            Number of frames.
        frame_size : tuple(int, int)
            Frame size.

        Returns
        -------
        WindowIndex
            Desired window index.
        """
        return calc_serial_window_iterator_index(
            length=video_length,
            window_size=RAFTIterator._calc_window_size(frame_size),
            padding=(1, 0),
            edge_mode="trim")

    @staticmethod
    def _calc_window_size(frame_size: tuple[int, int]) -> int:
        """
        Calculate window size.

        Parameters
        ----------
        frame_size : tuple(int, int)
            Frame size.

        Returns
        -------
        int
            Desired window size.
        """
        assert (frame_size[0] > 0)
        assert (frame_size[1] > 0)
        max_frame_size = max(frame_size[0], frame_size[1])
        if max_frame_size <= 640:
            window_size = 12
        elif max_frame_size <= 720:
            window_size = 8
        elif max_frame_size <= 1280:
            window_size = 4
        elif max_frame_size <= 1980:
            window_size = 2
        else:
            window_size = 1
        return window_size


def _test():
    raft_model_path = "../../../pytorchcv_data/test/raft-things_2.pth"

    time = 140
    height = 240
    width = 432
    frames = torch.randn(time, 3, height, width)

    flow_loader = RAFTIterator(
        frames=frames,
        raft_model_path=raft_model_path)

    video_length = time
    time_step = 10
    flow_loader_trim_pad = 3
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        flows_i = flow_loader[s:e]
        assert (flows_i is not None)
        flow_loader.trim_buffer_to(max(e - flow_loader_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
