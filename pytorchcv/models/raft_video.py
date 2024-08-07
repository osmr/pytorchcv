"""
    Video optical flow calculator based on RAFT.
"""

import os
import cv2
from PIL import Image
import scipy.ndimage
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from enum import IntEnum
from .raft import raft_things, calc_bidirectional_optical_flow_on_video_by_raft
from .propainter_rfc import propainter_rfc, calc_bidirectional_opt_flow_completion_by_pprfc
from .propainter_ip import propainter_ip
from .propainter import propainter
from .raft_stream import RAFTDataLoader


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
    data : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
        Data.
    """
    def __init__(self,
                 data: tuple[Any, ...] | list[Any] | Any):
        super(BufferedDataLoader, self).__init__()
        if isinstance(data, (tuple, list)):
            assert (len(data) > 0)
            self.raw_data_list = data
        else:
            self.raw_data_list = [data]

        self.start_pos = 0
        self.end_pos = 0
        # self.pos = 0
        self.buffer = None

    def __len__(self):
        return len(self.raw_data_list[0])

    def _load_data_items(self,
                         raw_data_chunk_list: tuple[Any, ...] | list[Any] | Any):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
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

        self.image_scaled_size = None
        self.do_scale = False

    def _rescale_image(self,
                       image: Image,
                       resample: IntEnum | None = None) -> Image:
        """
        Rescale frame.

        Parameters
        ----------
        image : Image
            Frame.
        resample : IntEnum or None, default None
            PIL resample mode.

        Returns
        -------
        Image
            Image.
        """
        if self.image_scaled_size is None:
            image_raw_size = image.size
            self.image_scaled_size = (int(self.image_resize_ratio * image_raw_size[0]),
                                      int(self.image_resize_ratio * image_raw_size[1]))
            self.image_scaled_size = (self.image_scaled_size[0] - self.image_scaled_size[0] % 8,
                                      self.image_scaled_size[1] - self.image_scaled_size[1] % 8)
            if image_raw_size != self.image_scaled_size:
                self.do_scale = True
        if self.do_scale:
            image = image.resize(
                size=self.image_scaled_size,
                resample=resample)
        return image

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
        frame = self._rescale_image(image=frame)
        return frame

    def _load_data_items(self,
                         raw_data_chunk_list: tuple[Any, ...] | list[Any] | Any):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Resulted data.
        """
        assert (len(raw_data_chunk_list) == 1)

        frame_list = [self.load_frame(x) for x in raw_data_chunk_list[0]]
        frames = np.stack(frame_list)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = frames.float()
        frames = frames.div(255.0)
        frames = frames * 2.0 - 1.0

        if self.use_cuda:
            frames = frames.cuda()

        return frames

    def _expand_buffer_by(self,
                          data_chunk: Any):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk])


class MaskBufferedDataLoader(FrameBufferedDataLoader):
    """
    Mask buffered data loader.

    Parameters
    ----------
    mask_dilation : int
        Mask dilation.
    image_resize_ratio : float
        Resize ratio.
    use_cuda : bool
        Whether to use CUDA.
    """
    def __init__(self,
                 mask_dilation: int,
                 **kwargs):
        super(MaskBufferedDataLoader, self).__init__(**kwargs)
        self.mask_dilation = mask_dilation
        assert (self.mask_dilation > 0)

    def load_mask(self,
                  mask_path: str) -> Image:
        """
        Load mask from file.

        Parameters
        ----------
        mask_path : str
            Path to mask file.

        Returns
        -------
        Image
            Mask.
        """
        mask = Image.open(mask_path)
        mask = self._rescale_image(image=mask, resample=Image.NEAREST)
        mask = np.array(mask.convert("L"))

        mask = scipy.ndimage.binary_dilation(input=mask, iterations=self.mask_dilation).astype(np.uint8)
        mask = Image.fromarray(mask * 255)

        return mask

    def _load_data_items(self,
                         raw_data_chunk_list: tuple[Any, ...] | list[Any] | Any):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Resulted data.
        """
        assert (len(raw_data_chunk_list) == 1)

        mask_list = [self.load_mask(x) for x in raw_data_chunk_list[0]]
        masks = np.stack(mask_list)
        masks = np.expand_dims(masks, axis=-1)
        masks = torch.from_numpy(masks).permute(0, 3, 1, 2).contiguous()
        masks = masks.float()
        masks = masks.div(255.0)

        if self.use_cuda:
            masks = masks.cuda()

        return masks


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
                         raw_data_chunk_list: tuple[Any, ...] | list[Any] | Any):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
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
                          data_chunk: Any):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk], dim=0)


class FlowCompWindowBufferedDataLoader(WindowBufferedDataLoader):
    """
    Optical flow completion window buffered data loader.

    Parameters
    ----------
    net: nn.Module
        Optical flow completion model.
    """
    def __init__(self,
                 net: nn.Module,
                 **kwargs):
        super(FlowCompWindowBufferedDataLoader, self).__init__(**kwargs)
        self.net = net

    def _load_data_items(self,
                         raw_data_chunk_list: tuple[Any, ...] | list[Any] | Any):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Resulted data.
        """
        assert (len(raw_data_chunk_list) == 2)

        flows = raw_data_chunk_list[0]
        masks = raw_data_chunk_list[1]

        masks_forward = masks[:-1].contiguous()
        masks_backward = masks[1:].contiguous()
        flow_masks = torch.cat((masks_forward, masks_backward), dim=1)

        comp_flows, _ = calc_bidirectional_opt_flow_completion_by_pprfc(
            net=self.net,
            flows=flows,
            flow_masks=flow_masks)
        assert (len(comp_flows.shape) == 4)
        assert (comp_flows.shape[1] == 4)

        # torch.cuda.empty_cache()
        return comp_flows

    def _expand_buffer_by(self,
                          data_chunk: Any):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk], dim=0)


class ImagePropWindowBufferedDataLoader(WindowBufferedDataLoader):
    """
    Image propagation window buffered data loader.

    Parameters
    ----------
    net: nn.Module
        Image propagation model.
    """
    def __init__(self,
                 net: nn.Module,
                 **kwargs):
        super(ImagePropWindowBufferedDataLoader, self).__init__(**kwargs)
        self.net = net

    def _load_data_items(self,
                         raw_data_chunk_list: tuple[Any, ...] | list[Any] | Any):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
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
                          data_chunk: Any):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk], dim=0)


class ImageTransWindowBufferedDataLoader(WindowBufferedDataLoader):
    """
    Image transformer window buffered data loader.

    Parameters
    ----------
    net: nn.Module
        Video inpainting model.
    """
    def __init__(self,
                 net: nn.Module,
                 stride,
                 ref_stride,
                 num_refs,
                 **kwargs):
        super(ImageTransWindowBufferedDataLoader, self).__init__(**kwargs)
        self.net = net
        self.stride = stride
        self.ref_stride = ref_stride
        self.num_refs = num_refs

    def _load_data_items(self,
                         raw_data_chunk_list: tuple[Any, ...] | list[Any] | Any):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Resulted data.
        """
        assert (len(raw_data_chunk_list) == 3)

        updated_frames_masks = raw_data_chunk_list[0]
        masks = raw_data_chunk_list[1]
        comp_flows = raw_data_chunk_list[2]

        prop_frames, updated_masks = torch.split(updated_frames_masks, [3, 1], dim=1)

        win_pos = self.window_pos + 1
        s_idx = win_pos * self.stride

        neighbor_ids = ImageTransWindowBufferedDataLoader._calc_image_trans_neighbor_index(
            mid_neighbor_id=s_idx,
            length=self.length,
            neighbor_stride=self.stride)
        ref_ids = ImageTransWindowBufferedDataLoader._calc_image_trans_ref_index(
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
                          data_chunk: Any):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
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


class VideoInpaintBufferedDataLoader(BufferedDataLoader):
    """
    Video inpainting buffered data loader.
    """
    def __init__(self,
                 **kwargs):
        super(VideoInpaintBufferedDataLoader, self).__init__(**kwargs)

    def _load_data_items(self,
                         raw_data_chunk_list: tuple[Any, ...] | list[Any] | Any):
        """
        Load data items.

        Parameters
        ----------
        raw_data_chunk_list : tuple(protocol(any), ...) or list(protocol(any)) or protocol(any)
            Raw data chunk.

        Returns
        -------
        protocol(any)
            Resulted data.
        """
        assert (len(raw_data_chunk_list) == 3)

        pred_frames = raw_data_chunk_list[0]
        frames = raw_data_chunk_list[1]
        masks = raw_data_chunk_list[2]

        pred_frames = pred_frames * masks + frames * (1 - masks)

        pred_frames = (((pred_frames + 1.0) / 2.0) * 255).to(torch.uint8)
        pred_frames = pred_frames.permute(0, 2, 3, 1).cpu().detach().numpy()

        return pred_frames

    def _expand_buffer_by(self,
                          data_chunk: Any):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : protocol(any)
            Data chunk.
        """
        self.buffer = np.concatenate([self.buffer, data_chunk])


def check_arrays(gt_arrays_dir_path,
                 pref,
                 tested_array,
                 start_idx,
                 end_idx,
                 c_slice=slice(None),
                 do_save=False,
                 precise=True,
                 atol: float = 1.0):
    if do_save and (not os.path.exists(gt_arrays_dir_path)):
        os.mkdir(gt_arrays_dir_path)

    for j, i in enumerate(range(start_idx, end_idx)):
        if isinstance(tested_array, torch.Tensor):
            tested_array_i = tested_array[j, c_slice].cpu().detach().numpy()
        else:
            tested_array_i = tested_array[j]

        tested_array_i_file_path = os.path.join(gt_arrays_dir_path, pref + "{:05d}.npy".format(i))
        if do_save:
            np.save(tested_array_i_file_path, np.ascontiguousarray(tested_array_i))
            continue

        gt_array_i = np.load(tested_array_i_file_path)

        if precise:
            if not np.array_equal(tested_array_i, gt_array_i):
                print(f"{gt_arrays_dir_path}, {pref}, {tested_array}, {start_idx}, {end_idx}, {j}, {i}")
            np.testing.assert_array_equal(tested_array_i, gt_array_i)
        else:
            if not np.allclose(tested_array_i, gt_array_i, rtol=0, atol=atol):
                print(f"{gt_arrays_dir_path}, {pref}, {tested_array}, {start_idx}, {end_idx}, {j}, {i}")
            np.testing.assert_allclose(tested_array_i, gt_array_i, rtol=0, atol=atol)


def run_streaming_propainter(frames_dir_path: str,
                             masks_dir_path: str,
                             image_resize_ratio: float,
                             raft_model_path: str,
                             pprfc_model_path: str,
                             pp_model_path: str,
                             mask_dilation: int = 4,
                             raft_iters: int = 20) -> np.ndarray:
    """
    Run ProPainter in streaming mode.

    Parameters
    ----------
    frames_dir_path : str
        Frames directory path.
    masks_dir_path : str
        Masks directory path.
    image_resize_ratio : float
        Resize ratio.
    raft_model_path : str
        Path to RAFT model parameters.
    pprfc_model_path : str
        Path to ProPainter-RFC model parameters.
    pp_model_path : str
        Path to ProPainter model parameters.
    use_cuda : bool, default True
        Whether to use CUDA.
    mask_dilation : int, default 4
        Mask dilation.
    raft_iters : int, default 20
        Number of iterations in RAFT.

    Returns
    -------
    np.ndarray
        Resulted frames.
    """
    pprfc_net = propainter_rfc()
    pprfc_net.load_state_dict(torch.load(pprfc_model_path, map_location="cpu", weights_only=True))
    for p in pprfc_net.parameters():
        p.requires_grad = False
    pprfc_net.eval()
    pprfc_net = pprfc_net.cuda()

    ppip_net = propainter_ip()
    ppip_net.eval()
    ppip_net = ppip_net.cuda()

    pp_net = propainter()
    pp_net.load_state_dict(torch.load(pp_model_path, map_location="cpu", weights_only=True))
    for p in pp_net.parameters():
        p.requires_grad = False
    pp_net.eval()
    pp_net = pp_net.cuda()

    frame_loader = FrameBufferedDataLoader(
        data=FilePathDirIterator(frames_dir_path),
        image_resize_ratio=image_resize_ratio,
        use_cuda=True)

    mask_loader = MaskBufferedDataLoader(
        mask_dilation=mask_dilation,
        data=FilePathDirIterator(masks_dir_path),
        image_resize_ratio=image_resize_ratio,
        use_cuda=True)

    video_length = len(frame_loader)

    flow_loader = RAFTDataLoader(
        frame_loader=frame_loader,
        raft_model_path=raft_model_path,
        raft_iters=raft_iters)

    pprfc_flows_window_index = calc_window_data_loader_index(
        length=video_length - 1,
        window_size=80,
        padding=(5, 5),
        edge_mode="ignore")
    pprfc_mask_window_index = calc_window_data_loader_index(
        length=video_length,
        window_size=80,
        padding=(5, 6),
        edge_mode="ignore")
    pprfc_window_index = cat_window_data_loader_indices([pprfc_flows_window_index, pprfc_mask_window_index])

    flow_comp_loader = FlowCompWindowBufferedDataLoader(
        data=[flow_loader, mask_loader],
        window_index=pprfc_window_index,
        net=pprfc_net)

    ppip_images_window_index = calc_window_data_loader_index(
        length=video_length,
        window_size=80,
        padding=(10, 10),
        edge_mode="ignore")
    ppip_flows_window_index = calc_window_data_loader_index(
        length=video_length - 1,
        window_size=80,
        padding=(10, 9),
        edge_mode="ignore")
    ppip_window_index = cat_window_data_loader_indices([ppip_images_window_index, ppip_images_window_index,
                                                        ppip_flows_window_index])

    image_prop_loader = ImagePropWindowBufferedDataLoader(
        data=[frame_loader, mask_loader, flow_comp_loader],
        window_index=ppip_window_index,
        net=ppip_net)

    pp_stride = 5
    pp_ref_stride = 10
    pp_ref_window_size = 80
    pp_num_refs = pp_ref_window_size // pp_ref_stride
    pp_ref_frames_window_index = calc_window_data_loader_index2(
        length=video_length,
        stride=pp_stride,
        src_padding=(40, 41),
        padding=(5, 6))
    pp_local_flows_window_index = calc_window_data_loader_index2(
        length=video_length,
        stride=pp_stride,
        src_padding=(5, 5),
        padding=(5, 6))
    ppip_window_index = cat_window_data_loader_indices([pp_ref_frames_window_index, pp_ref_frames_window_index,
                                                        pp_local_flows_window_index])

    image_trans_loader = ImageTransWindowBufferedDataLoader(
        data=[image_prop_loader, mask_loader, flow_comp_loader],
        window_index=ppip_window_index,
        net=pp_net,
        stride=pp_stride,
        ref_stride=pp_ref_stride,
        num_refs=pp_num_refs)

    video_inpaint_loader = VideoInpaintBufferedDataLoader(
        data=[image_trans_loader, frame_loader, mask_loader])

    vi_step = 10
    image_trans_loader_trim_pad = 6
    image_prop_loader_trim_pad = 35
    flow_comp_loader_trim_pad = 3
    flow_loader_trim_pad = 3
    mask_loader_trim_pad = 35
    frame_loader_trim_pad = 2
    for s in range(0, video_length, vi_step):
        e = min(s + vi_step, video_length)
        vi_frames_i = video_inpaint_loader[s:e]
        image_trans_loader.trim_buffer_to(max(e - image_trans_loader_trim_pad, 0))
        image_prop_loader.trim_buffer_to(max(e - image_prop_loader_trim_pad, 0))
        flow_comp_loader.trim_buffer_to(max(e - flow_comp_loader_trim_pad, 0))
        flow_loader.trim_buffer_to(max(e - flow_loader_trim_pad, 0))
        mask_loader.trim_buffer_to(max(e - mask_loader_trim_pad, 0))
        frame_loader.trim_buffer_to(max(e - frame_loader_trim_pad, 0))
        torch.cuda.empty_cache()

    assert (video_inpaint_loader.start_pos == 0)
    return video_inpaint_loader.buffer


def _test():
    # root_path = "../../../pytorchcv_data/test0"
    # image_resize_ratio = 1.0
    # video_length = 80

    root_path = "../../../pytorchcv_data/test1"
    image_resize_ratio = 0.5
    video_length = 287

    frames_dir_name = "_source_frames"
    masks_dir_name = "_segmentation_masks"
    frames_dir_path = os.path.join(root_path, frames_dir_name)
    masks_dir_path = os.path.join(root_path, masks_dir_name)

    raft_model_path = "../../../pytorchcv_data/test/raft-things_2.pth"
    pprfc_model_path = "../../../pytorchcv_data/test/propainter_rfc.pth"
    pp_model_path = "../../../pytorchcv_data/test/propainter.pth"

    vi_frames = run_streaming_propainter(
        frames_dir_path=frames_dir_path,
        masks_dir_path=masks_dir_path,
        image_resize_ratio=image_resize_ratio,
        raft_model_path=raft_model_path,
        pprfc_model_path=pprfc_model_path,
        pp_model_path=pp_model_path)

    video_length = len(vi_frames)

    if True:
        comp_frames_dir_path = os.path.join(root_path, "comp_frames")
        check_arrays(
            gt_arrays_dir_path=comp_frames_dir_path,
            pref="comp_frame_",
            tested_array=vi_frames,
            start_idx=0,
            end_idx=video_length,
            # do_save=True,
            precise=False,
            # atol=8,
        )

    pass


if __name__ == "__main__":
    _test()
