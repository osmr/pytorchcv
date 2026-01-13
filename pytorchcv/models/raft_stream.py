"""
    Streaming mode of RAFT, implemented in PyTorch.
    Original paper: 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,'
    https://arxiv.org/pdf/2003.12039.
"""

__all__ = ['RAFTSequencer']

import torch
import torch.nn as nn
from typing import Sequence
from pytorchcv.models.common.stream import WindowBufferedSequencer, WindowIndex, calc_serial_window_sequencer_index
from pytorchcv.models.raft import raft_things, calc_bidirectional_optical_flow_on_video_by_raft


class RAFTSequencer(WindowBufferedSequencer):
    """
    Optical flow calculation (RAFT) window buffered sequencer.

    Parameters
    ----------
    frames : sequence
        Frame sequencer.
    raft_model : nn.Module or str or None, default None
        RAFT model or path to RAFT model parameters.
    use_cuda : bool, default True
        Whether to use CUDA.
    raft_iters : int, default 20
        Number of iterations in RAFT.
    window_size : int or None, default None
        Window size.
    """
    def __init__(self,
                 frames: Sequence,
                 raft_model: nn.Module | str | None = None,
                 use_cuda: bool = True,
                 raft_iters: int = 20,
                 window_size: int | None = None,
                 **kwargs):
        assert (len(frames) > 1)
        super(RAFTSequencer, self).__init__(
            data=frames,
            window_index=RAFTSequencer._calc_window_index(
                video_length=len(frames),
                window_size=window_size,
                frame_size=frames[0].shape[1:]),
            **kwargs)
        self.net = RAFTSequencer._load_model(
            raft_model=raft_model,
            use_cuda=use_cuda,
            raft_iters=raft_iters)

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

    @staticmethod
    def _load_model(raft_model: nn.Module | str | None = None,
                    use_cuda: bool = True,
                    raft_iters: int = 20) -> nn.Module:
        """
        Load RAFT model.

        Parameters
        ----------
        raft_model : nn.Module or str or None, default None
            RAFT model or path to RAFT model parameters.
        use_cuda : bool, default True
            Whether to use CUDA.
        raft_iters : int, default 20
            Number of iterations in RAFT.

        Returns
        -------
        nn.Module
            RAFT model.
        """
        if isinstance(raft_model, nn.Module):
            assert (raft_model.iters == raft_iters)
            return raft_model
        if raft_model is not None:
            raft_net = raft_things(
                in_normalize=False,
                iters=raft_iters)
            raft_net.load_state_dict(torch.load(raft_model, map_location="cpu", weights_only=True))
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
        if use_cuda:
            raft_net = raft_net.cuda()
        return raft_net

    @staticmethod
    def _calc_window_index(video_length: int,
                           window_size: int | None,
                           frame_size: tuple[int, int]) -> WindowIndex:
        """
        Calculate window index.

        Parameters
        ----------
        video_length : int
            Number of frames.
        window_size : int or None
            Window size.
        frame_size : tuple(int, int)
            Frame size.

        Returns
        -------
        WindowIndex
            Desired window index.
        """
        return calc_serial_window_sequencer_index(
            length=video_length,
            target_length=video_length,
            window_size=RAFTSequencer._calc_window_size(
                window_size=window_size,
                frame_size=frame_size),
            padding=(1, 0),
            edge_mode="trim")

    @staticmethod
    def _calc_window_size(
            window_size: int | None,
            frame_size: tuple[int, int]) -> int:
        """
        Calculate window size.

        Parameters
        ----------
        window_size : int or None
            Window size.
        frame_size : tuple(int, int)
            Frame size.

        Returns
        -------
        int
            Desired window size.
        """
        if window_size is not None:
            assert (window_size > 0)
            return window_size

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
    time = 14
    height = 240
    width = 432
    frames = torch.randn(time, 3, height, width).cuda()

    flow_sequencer = RAFTSequencer(
        frames=frames,
        raft_model=None,
        use_cuda=True)

    video_length = time
    time_step = 10
    flow_sequencer_trim_pad = 2
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        flows_i = flow_sequencer[s:e]
        assert (flows_i is not None)
        flow_sequencer.trim_buffer_to(max(e - flow_sequencer_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
