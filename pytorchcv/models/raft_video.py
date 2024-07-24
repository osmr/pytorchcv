"""
    Video optical flow calculator based on RAFT.
"""

import os
import cv2
from PIL import Image
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Protocol
from raft import raft_things


class FilePathDirIterator(object):
    """
    Iterator for file paths in directory.

    Parameters
    ----------
    dir_path str
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
            Resulted data.
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


class FrameBufferedDataLoader(BufferedDataLoader):
    """
    Buffered data loader.

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


def calc_window_data_loader_index(length: int,
                                  window_size: int = 1,
                                  stride: int = 1,
                                  in_padding: tuple[int, int] = (0, 0),
                                  reduce: bool = True,
                                  reduce_first_window: bool = False) -> OrderedDict[tuple[int, int], tuple[int, int]]:
    """
    Calculate window data loader index.

    Parameters
    ----------
    length : int
        Data length.
    window_size : int, default 1
        Calculation window size.
    stride : int, default 1
        Stride of the calculation.
    in_padding : tuple(int, int), default (0, 0)
        Internal padding (overlap) value for calculation.
    reduce : bool, default True
        Whether to reduce output size due to padding.
    reduce_first_window : bool, default False
        Whether to reduce size of the first window.

    Returns
    -------
    OrderedDict(tuple(int, int), tuple(int, int))
        Resulted index.
    """
    assert (length > 0)
    assert (window_size > 0)
    assert (stride > 0)
    assert (in_padding[0] >= 0) and (in_padding[1] >= 0)

    if reduce_first_window:
        out_index = [(0, min(length, window_size - 1))]
    else:
        in_start_index = list(range(0, length, window_size))
    index = OrderedDict()
    for s_idx in range(length):
        e_idx = min(length, s_idx + window_size - 1)
        s_idx_ext = s_idx if s_idx == 0 else s_idx - 1
        index[(s_idx_ext, e_idx)] = (s_idx_ext, e_idx)

    return index


class WindowDataLoaderIndexCalculator(object):
    """
    Window data loader index calculator.

    Parameters
    ----------
    length : int
        Data length.
    window_size : int, default 1
        Calculation window size.
    stride : int, default 1
        Stride of the calculation.
    in_padding : tuple(int, int), default (0, 0)
        Internal padding (overlap) value for calculation.
    reduce : bool, default True
        Whether to reduce output size due to padding.
    """
    def __init__(self,
                 length: int,
                 window_size: int = 1,
                 stride: int = 1,
                 in_padding: tuple[int, int] = (0, 0),
                 reduce: bool = True):
        super(WindowDataLoaderIndexCalculator, self).__init__()
        assert (length > 0)
        assert (window_size > 0)
        assert (stride > 0)
        assert (in_padding[0] >= 0) and (in_padding[1] >= 0)

        self.length = length
        self.window_size = window_size
        self.stride = stride
        self.in_padding = in_padding
        self.reduce = reduce

        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.data[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

    def __len__(self):
        try:
            return self.data.__len__()
        except AttributeError:
            return sum(1 for item in iter(self.data))

    def __getitem__(self, index):
        return self.data[index]


class BufferedCalculator(object):
    """
    Abstract buffered calculator.

    Parameters
    ----------
    data : protocol(any)
        Data.
    window_size : int, default 1
        Calculation window size.
    stride : int, default 1
        Stride of the calculation.
    in_padding : tuple(int, int), default (0, 0)
        Internal padding (overlap) value for calculation.
    """
    def __init__(self,
                 data,
                 window_size: int = 1,
                 stride: int = 1,
                 in_padding: tuple[int, int] = (0, 0)):
        super(BufferedCalculator, self).__init__()
        assert (window_size > 0)
        assert (stride > 0)
        assert (in_padding[0] >= 0) and (in_padding[1] >= 0)

        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.in_padding = in_padding

        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.data[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

    def __len__(self):
        try:
            return self.data.__len__()
        except AttributeError:
            return sum(1 for item in iter(self.data))

    def __getitem__(self, index):
        return self.data[index]


def _test():
    root_path = "../../../pytorchcv_data/test0"
    frames_dir_name = "_source_frames"
    frames_dir_path = os.path.join(root_path, frames_dir_name)
    image_resize_ratio = 1.0
    use_cuda = True

    # loader = BufferedDataLoader(data=FilePathDirIterator(frames_dir_path))
    loader = FrameBufferedDataLoader(data=FilePathDirIterator(frames_dir_path), image_resize_ratio=image_resize_ratio, use_cuda=use_cuda)
    # a = loader[:]
    # a = loader[2:]
    a = loader[:10]
    a = loader[0:10]
    a = loader[7]
    a = loader[0:20]
    a = loader[10:20]
    a = loader[21:30]
    loader.trim_buffer_to(25)
    a = loader[26:31]
    loader.trim_buffer_to(26)
    a = loader[27:110]
    pass


if __name__ == "__main__":
    _test()
