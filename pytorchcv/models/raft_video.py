"""
    Video optical flow calculator based on RAFT.
"""

import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from raft import raft_things, run_raft_on_video


class FilePathDirIterator(object):
    """
    Iterator for file paths in directory.

    Parameters
    ----------
    dir_path: str
        Directory path.
    """
    def __init__(self,
                 dir_path: str):
        super(FilePathDirIterator, self).__init__()
        assert os.path.exists(dir_path)

        self.dir_path = dir_path
        self.file_name_list = sorted(os.listdir(dir_path))

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self,
                    index: int | slice):
        selected_file_name_list = self.file_name_list[index]
        if isinstance(selected_file_name_list, str):
            return os.path.join(self.dir_path, selected_file_name_list)
        elif isinstance(selected_file_name_list, list):
            return [os.path.join(self.dir_path, x) for x in selected_file_name_list]
        else:
            raise ValueError()


class BufferedDataLoader(object):
    """
    Buffered data loader.

    Parameters
    ----------
    data : protocol(any)
        Data.
    """
    def __init__(self,
                 data):
        super(BufferedDataLoader, self).__init__()
        self.raw_data = data

        self.start_pos = 0
        self.end_pos = 0
        self.pos = 0
        self.buffer = None

    def __len__(self):
        return len(self.raw_data)

    def _load_data_items(self,
                         raw_data_chunk):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk : protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Target data.
        """
        return raw_data_chunk

    def _expand_buffer_by(self,
                          data_chunk):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
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
        raw_data_chunk = self.raw_data[self.end_pos:end]
        if self.buffer is None:
            self.buffer = self._load_data_items(raw_data_chunk)
        else:
            data_chunk = self._load_data_items(raw_data_chunk)
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

        return self.buffer[index]

    def trim_buffer_to(self,
                       start):
        """
        Trim buffer.

        Parameters
        ----------
        start : int
            Start index for saved buffer.
        """
        assert (start >= 0)
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
    """
    def __init__(self,
                 target: WindowRange,
                 source: WindowRange):
        super(WindowMap, self).__init__()
        self.target = target
        self.source = source

    def __repr__(self):
        return "{target} <- {source}".format(
            target=self.target,
            source=self.source)


WindowIndex = list[WindowMap]


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
        assert (e > s)
        index.append(WindowMap(
            target=WindowRange(start=s, stop=e),
            source=WindowRange(start=src_s, stop=src_e)))
    return index


class WindowBufferedDataLoader(BufferedDataLoader):
    """
    Window buffered data loader.

    Parameters
    ----------
    data : protocol(any)
        Data.
    """
    def __init__(self,
                 window_index: WindowIndex,
                 **kwargs):
        super(WindowBufferedDataLoader, self).__init__(**kwargs)
        self.window_index = window_index

        self.length = self.window_index[-1].target.stop
        self.window_length = len(self.window_index)
        self.window_pose = -1

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
        for win_pose in range(max(self.window_pose, 0), self.window_length):
            win_stop = self.window_index[win_pose].target.stop
            if pos <= win_stop:
                return win_pose
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
        for win_pose in range(max(self.window_pose, 0), win_end + 1):
            win_map = self.window_index[win_pose]
            raw_data_chunk = self.raw_data[win_map.source.start:win_map.source.stop]
            if self.buffer is None:
                self.buffer = self._load_data_items(raw_data_chunk)
            else:
                data_chunk = self._load_data_items(raw_data_chunk)
                self._expand_buffer_by(data_chunk)
            self.end_pos = win_map.target.stop


class FrameBufferedDataLoader(BufferedDataLoader):
    """
    Frame buffered data loader.

    Parameters
    ----------
    image_resize_ratio : float
        Resize ratio.
    use_cuda : bool
        Whether to use CUDA.
    """
    def __init__(self,
                 image_resize_ratio: float,
                 use_cuda: bool,
                 **kwargs):
        super(FrameBufferedDataLoader, self).__init__(**kwargs)
        assert (image_resize_ratio > 0.0)
        self.image_resize_ratio = image_resize_ratio
        self.use_cuda = use_cuda

        self.frame_scaled_size = None
        self.do_scale = False

    def load_frame(self,
                   frame_path: str) -> Image:
        """
        Load frame from file.

        Parameters
        ----------
        frame_path : str
            Path to frame file.

        Returns
        -------
        Image
            Frame.
        """
        frame = cv2.imread(frame_path)
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.frame_scaled_size is None:
            frame_raw_size = frame.size
            self.frame_scaled_size = (int(self.image_resize_ratio * frame_raw_size[0]),
                                      int(self.image_resize_ratio * frame_raw_size[1]))
            self.frame_scaled_size = (self.frame_scaled_size[0] - self.frame_scaled_size[0] % 8,
                                      self.frame_scaled_size[1] - self.frame_scaled_size[1] % 8)
            if frame_raw_size != self.frame_scaled_size:
                self.do_scale = True
        if self.do_scale:
            frame = frame.resize(self.frame_scaled_size)
        return frame

    def _load_data_items(self,
                         raw_data_chunk):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk : protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Resulted data.
        """
        frame_list = [self.load_frame(x) for x in raw_data_chunk]
        frames = np.stack(frame_list)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = frames.float()
        frames = frames.div(255.0)
        frames = frames * 2.0 - 1.0

        if self.use_cuda:
            frames = frames.cuda()

        return frames

    def _expand_buffer_by(self,
                          data_chunk):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk])


class FlowWindowBufferedDataLoader(WindowBufferedDataLoader):
    """
    Optical flow window buffered data loader.

    Parameters
    ----------
    net: nn.Module
        Optical flow model.
    frames_loader : FrameBufferedDataLoader
        Frames.
    iters : int, default 20
        Number of iterations for flow calculation.
    """
    def __init__(self,
                 net: nn.Module,
                 iters: int = 20,
                 **kwargs):
        super(FlowWindowBufferedDataLoader, self).__init__(**kwargs)
        assert (iters > 0)

        self.net = net
        self.iters = iters

    def _load_data_items(self,
                         raw_data_chunk):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk : protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Resulted data.
        """
        flows = run_raft_on_video(
            net=self.net,
            frames=raw_data_chunk,
            iters=self.iters)
        # torch.cuda.empty_cache()
        return flows

    def _expand_buffer_by(self,
                          data_chunk):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk], dim=1)


def _test():
    root_path = "../../../pytorchcv_data/test0"
    frames_dir_name = "_source_frames"
    frames_dir_path = os.path.join(root_path, frames_dir_name)
    image_resize_ratio = 1.0
    use_cuda = True
    raft_iters = 20
    raft_model_path = root_path = "../../../pytorchcv_data/test/raft-things_2.pth"

    # ind1 = calc_window_data_loader_index(
    #     length=876,
    #     window_size=80,
    #     padding=(5, 5),
    #     edge_mode="ignore")
    #
    # ind1 = calc_window_data_loader_index(
    #     length=877,
    #     window_size=12,
    #     padding=(1, 0),
    #     edge_mode="trim")
    #
    # ind1 = calc_window_data_loader_index(
    #     length=287,
    #     window_size=12,
    #     padding=(1, 0),
    #     edge_mode="trim")
    #
    # ind1 = calc_window_data_loader_index(
    #     length=80,
    #     window_size=12,
    #     padding=(1, 0),
    #     edge_mode="trim")

    # loader = BufferedDataLoader(data=FilePathDirIterator(frames_dir_path))
    frames_loader = FrameBufferedDataLoader(
        data=FilePathDirIterator(frames_dir_path),
        image_resize_ratio=image_resize_ratio,
        use_cuda=use_cuda)

    raft_net = raft_things()
    raft_net.load_state_dict(torch.load(raft_model_path, map_location="cpu"))
    for p in raft_net.parameters():
        p.requires_grad = False
    raft_net.eval()
    raft_net = raft_net.cuda()

    raft_window_index = calc_window_data_loader_index(
        length=80,
        window_size=12,
        padding=(1, 0),
        edge_mode="trim")

    raft_loader = FlowWindowBufferedDataLoader(
        data=frames_loader,
        window_index=raft_window_index,
        net=raft_net,
        iters=raft_iters)
    a = raft_loader[:]
    # a = raft_loader[raft_window_index[0].target.start:raft_window_index[0].target.stop]

    # # a = loader[:]
    # # a = loader[2:]
    # a = loader[:10]
    # a = loader[0:10]
    # a = loader[7]
    # a = loader[0:20]
    # a = loader[10:20]
    # a = loader[21:30]
    # loader.trim_buffer_to(25)
    # a = loader[26:31]
    # loader.trim_buffer_to(26)
    # a = loader[27:110]
    pass


if __name__ == "__main__":
    _test()
