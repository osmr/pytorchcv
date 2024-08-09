"""
    Streaming mode of RAFT, implemented in PyTorch.
    Original paper: 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,'
    https://arxiv.org/pdf/2003.12039.
"""

__all__ = ['RAFTDataLoader', 'WindowBufferedDataLoader', 'WindowIndex', 'WindowMultiIndex',
           'calc_window_data_loader_index', 'cat_window_data_loader_indices']

import torch
import torch.nn as nn
from typing import Sequence
from .raft import raft_things, calc_bidirectional_optical_flow_on_video_by_raft


class BufferedDataLoader(object):
    """
    Buffered data loader.

    Parameters
    ----------
    data : tuple(sequence, ...) or list(sequence) or sequence
        Data.
    """
    def __init__(self,
                 data: tuple[Sequence, ...] | list[Sequence] | Sequence):
        super(BufferedDataLoader, self).__init__()
        if isinstance(data, (tuple, list)):
            assert (len(data) > 0)
            self.raw_data_list = data
        else:
            self.raw_data_list = [data]

        self.start_pos = 0
        self.end_pos = 0
        self.buffer = None

    def __len__(self):
        return len(self.raw_data_list[0])

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
            Target data.
        """
        if len(raw_data_chunk_list) == 1:
            return raw_data_chunk_list
        else:
            raise Exception()

    def _expand_buffer_by(self,
                          data_chunk: Sequence):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : sequence
            Data chunk.
        """
        self.buffer += data_chunk

    def _expand_buffer_to(self,
                          end: int):
        """
        Expand buffer to the end index.

        Parameters
        ----------
        end : int
            End index.
        """
        assert (end > self.end_pos)
        raw_data_chunk_list = [raw_data[self.end_pos:end] for raw_data in self.raw_data_list]
        if self.buffer is None:
            self.buffer = self._load_data_items(raw_data_chunk_list)
        else:
            data_chunk = self._load_data_items(raw_data_chunk_list)
            self._expand_buffer_by(data_chunk)
        self.end_pos = end

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            end = index.stop
        elif isinstance(index, int):
            end = index + 1
        else:
            raise ValueError()

        if end is None:
            end = len(self)

        if end > self.end_pos:
            self._expand_buffer_to(end=end)

        if isinstance(index, slice):
            if self.start_pos > 0:
                new_start = index.start - self.start_pos if index.start is not None else None
                new_stop = index.stop - self.start_pos if index.stop is not None else None
                index = slice(new_start, new_stop, index.step)
        elif isinstance(index, int):
            index -= self.start_pos
        else:
            raise ValueError()

        if isinstance(index, slice):
            if not ((index.start is None) or (index.start >= 0)):
                pass
            assert (index.start is None) or (index.start >= 0)
            assert (index.stop is None) or (index.stop >= 0)
        elif isinstance(index, int):
            assert (index >= 0)
        else:
            raise ValueError()

        return self.buffer[index]

    def trim_buffer_to(self,
                       start: int):
        """
        Trim buffer.

        Parameters
        ----------
        start : int
            Start index for saved buffer.
        """
        assert (start >= 0)
        if not (start < self.end_pos - 1):
            pass
        assert (start < self.end_pos - 1)

        if start > self.start_pos:
            assert (self.buffer is not None)
            s_idx = start - self.start_pos
            self.buffer = self.buffer[s_idx:]
            self.start_pos = start


class WindowRange(object):
    """
    Window range.

    Parameters
    ----------
    start: int
        Start position.
    stop: int
        Stop position.
    """
    def __init__(self,
                 start: int,
                 stop: int):
        super(WindowRange, self).__init__()
        self.start = start
        self.stop = stop

    def __repr__(self):
        return "{start}:{stop}".format(
            start=self.start,
            stop=self.stop)


class WindowMap(object):
    """
    Window map.

    Parameters
    ----------
    target: WindowRange
        Target window range.
    source: WindowRange
        Source window range.
    target_start: int
        Start position for target.
    """
    def __init__(self,
                 target: WindowRange,
                 source: WindowRange,
                 target_start: int):
        super(WindowMap, self).__init__()
        self.target = target
        self.source = source
        self.target_start = target_start

    def __repr__(self):
        return "{target}:{target_start} <- {source}".format(
            target=self.target,
            source=self.source,
            target_start=self.target_start)


class WindowMultiMap(object):
    """
    Window multimap.

    Parameters
    ----------
    target: WindowRange
        Target window range.
    sources: list(WindowRange)
        Source window range.
    target_start: int
        Start position for target.
    """
    def __init__(self,
                 target: WindowRange,
                 sources: list[WindowRange],
                 target_start: int):
        super(WindowMultiMap, self).__init__()
        self.target = target
        self.sources = sources
        self.target_start = target_start

    def __repr__(self):
        s = "/".join(["{}".format(s) for s in self.sources])
        return "{target}:{target_start} <- {sources}".format(
            target=self.target,
            sources=s,
            target_start=self.target_start)


WindowIndex = list[WindowMap]
WindowMultiIndex = list[WindowMultiMap]


def calc_window_data_loader_index(length: int,
                                  window_size: int = 1,
                                  padding: tuple[int, int] = (0, 0),
                                  edge_mode: str = "ignore") -> WindowIndex:
    """
    Calculate window data loader index.

    Parameters
    ----------
    length : int
        Data length.
    window_size : int, default 1
        Calculation window size.
    padding : tuple(int, int), default (0, 0)
        Padding (overlap) values for raw data.
    edge_mode : str, options: 'ignore', 'trim', default 'ignore'
        Data edge processing mode:
        'ignore' - ignore padding on edges,
        'trim' - trim edges due to padding.

    Returns
    -------
    WindowIndex
        Resulted index.
    """
    assert (length > 0)
    assert (window_size > 0)
    assert (padding[0] >= 0) and (padding[1] >= 0)
    assert (edge_mode in ("ignore", "trim"))

    trim_values = padding if edge_mode == "trim" else (0, 0)
    index = []
    for i in range(0, length, window_size):
        src_s = max(i - padding[0], 0)
        src_e = min(i + window_size + padding[1], length)
        s = max(i - trim_values[0], 0)
        e = min(i - trim_values[0] + window_size, length - trim_values[0] - trim_values[1])
        target_start = 0 if edge_mode == "trim" else (i if i - padding[0] < 0 else padding[0])
        assert (e > s)
        index.append(WindowMap(
            target=WindowRange(start=s, stop=e),
            source=WindowRange(start=src_s, stop=src_e),
            target_start=target_start))
    return index


def calc_window_data_loader_index2(length: int,
                                   stride: int = 1,
                                   src_padding: tuple[int, int] = (0, 1),
                                   padding: tuple[int, int] = (0, 1)) -> WindowIndex:
    """
    Calculate window data loader index (the second version).

    Parameters
    ----------
    length : int
        Data length.
    stride : int, default 1
        Stride value.
    src_padding : tuple(int, int), default (0, 1)
        Padding (overlap) values for source data.
    padding : tuple(int, int), default (0, 1)
        Padding (overlap) values for target data.

    Returns
    -------
    WindowIndex
        Resulted index.
    """
    assert (length > 0)
    assert (stride > 0)
    assert (src_padding[0] >= 0) and (src_padding[1] >= 0)
    assert (padding[0] >= 0) and (padding[1] >= 0)

    padding_diff = max((padding[1] - src_padding[1]), 0)

    index = []
    for i in range(0, length, stride):
        src_s = max(i - src_padding[0], 0)
        src_e = min(i + src_padding[1], length - padding_diff)
        assert (src_e > src_s)
        s = max(i - padding[0], 0)
        e = min(i + padding[1], length)
        assert (e > s)
        index.append(WindowMap(
            target=WindowRange(start=s, stop=e),
            source=WindowRange(start=src_s, stop=src_e),
            target_start=0))
    return index


def cat_window_data_loader_indices(indices: list[WindowIndex]) -> WindowMultiIndex:
    """
    Concatenate window data loader indices.

    Parameters
    ----------
    indices : list(WindowIndex)
        Indices.

    Returns
    -------
    WindowMultiIndex
        Resulted multiindex.
    """
    index = [WindowMultiMap(x[0].target, [y.source for y in x], x[0].target_start) for x in zip(*indices)]
    return index


class WindowBufferedDataLoader(BufferedDataLoader):
    """
    Window buffered data loader.

    Parameters
    ----------
    window_index : WindowIndex or WindowMultiIndex
        Window index.
    """
    def __init__(self,
                 window_index: WindowIndex | WindowMultiIndex,
                 **kwargs):
        super(WindowBufferedDataLoader, self).__init__(**kwargs)
        assert (len(window_index) > 0)
        if isinstance(window_index[0], WindowMap):
            self.window_index = cat_window_data_loader_indices([window_index])
        else:
            self.window_index = window_index

        assert (len(self.raw_data_list) == len(self.window_index[0].sources))

        self.length = self.window_index[-1].target.stop
        self.window_length = len(self.window_index)
        self.window_pos = -1

    def __len__(self):
        return self.length

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
            win_stop = self.window_index[win_pos].target.stop
            if pos <= win_stop:
                return win_pos
        return self.window_length - 1

    def _expand_buffer_to(self,
                          end: int):
        """
        Expand buffer to the end index.

        Parameters
        ----------
        end : int
            End index.
        """
        assert (end > self.end_pos)
        win_end = self._calc_window_pose(end)
        for win_pos in range(max(self.window_pos + 1, 0), win_end + 1):
            win_mmap = self.window_index[win_pos]
            raw_data_chunk_list = [r_data[map_s.start:map_s.stop] for r_data, map_s in
                                   zip(self.raw_data_list, win_mmap.sources)]
            data_chunk = self._load_data_items(raw_data_chunk_list)

            # data_chunk = data_chunk[(win_mmap.target.start - self.end_pos):(win_mmap.target.stop - self.end_pos)]

            # assert (win_mmap.target.start - self.end_pos == 0)
            # data_chunk = data_chunk[win_mmap.target_start:(win_mmap.target.stop - self.end_pos + win_mmap.target_start)]  # noqa

            # assert (win_mmap.target.start == self.end_pos)
            data_chunk = data_chunk[win_mmap.target_start:(win_mmap.target.stop - win_mmap.target.start + win_mmap.target_start)]  # noqa

            if self.buffer is None:
                self.buffer = data_chunk
            else:
                self._expand_buffer_by(data_chunk)
            self.end_pos = win_mmap.target.stop
            self.window_pos = win_pos


class FlowWindowBufferedDataLoader(WindowBufferedDataLoader):
    """
    Optical flow window buffered data loader.

    Parameters
    ----------
    net: nn.Module
        Optical flow model.
    """
    def __init__(self,
                 net: nn.Module,
                 **kwargs):
        super(FlowWindowBufferedDataLoader, self).__init__(**kwargs)
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
        assert (len(raw_data_chunk_list) == 1)

        frames = raw_data_chunk_list[0]
        flows = calc_bidirectional_optical_flow_on_video_by_raft(
            net=self.net,
            frames=frames)
        assert (len(flows.shape) == 4)
        assert (flows.shape[1] == 4)

        # torch.cuda.empty_cache()
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


class RAFTDataLoader(FlowWindowBufferedDataLoader):
    """
    RAFT window buffered data loader.

    Parameters
    ----------
    frame_loader : sequence
        Frame data loader.
    raft_model_path : str or None, default None
        Path to RAFT model parameters.
    raft_iters : int, default 20
        Number of iterations in RAFT.
    """
    def __init__(self,
                 frame_loader: Sequence,
                 raft_model_path: str | None = None,
                 raft_iters: int = 20,
                 **kwargs):
        assert (len(frame_loader) > 0) 
        super(RAFTDataLoader, self).__init__(
            data=frame_loader,
            window_index=RAFTDataLoader._calc_window_index(
                video_length=len(frame_loader),
                frame_size=frame_loader[0].shape[1:]),
            net=RAFTDataLoader._load_model(
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
        return calc_window_data_loader_index(
            length=video_length,
            window_size=RAFTDataLoader._calc_window_size(frame_size),
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

    flow_loader = RAFTDataLoader(
        frame_loader=frames,
        raft_model_path=raft_model_path)

    video_length = time
    time_step = 10
    flow_loader_trim_pad = 3
    for s in range(0, video_length, time_step):
        e = min(s + time_step, video_length)
        flows_i = flow_loader[s:e]
        flow_loader.trim_buffer_to(max(e - flow_loader_trim_pad, 0))
        torch.cuda.empty_cache()

    pass


if __name__ == "__main__":
    _test()
