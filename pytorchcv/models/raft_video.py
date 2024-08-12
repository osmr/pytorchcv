"""
    Video optical flow calculator based on RAFT.
"""

import os
import cv2
from PIL import Image
import scipy.ndimage
import numpy as np
import torch
from typing import Sequence
from enum import IntEnum
from .raft_stream import RAFTIterator, BufferedIterator
from .propainter_rfc_stream import PPRFCIterator
from .propainter_ip_stream import PPIPIterator
from .propainter_stream import ProPainterIterator


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
                    index: int | slice) -> list[str]:
        selected_file_name_list = self.file_name_list[index]
        if isinstance(selected_file_name_list, str):
            return os.path.join(self.dir_path, selected_file_name_list)
        elif isinstance(selected_file_name_list, list):
            return [os.path.join(self.dir_path, x) for x in selected_file_name_list]
        else:
            raise ValueError()


class FrameIterator(BufferedIterator):
    """
    Frame buffered iterator.

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
        super(FrameIterator, self).__init__(**kwargs)
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
                          data_chunk: Sequence):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : sequence
            Data chunk.
        """
        self.buffer = torch.cat([self.buffer, data_chunk])


class MaskIterator(FrameIterator):
    """
    Mask buffered iterator.

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
        super(MaskIterator, self).__init__(**kwargs)
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

        mask_list = [self.load_mask(x) for x in raw_data_chunk_list[0]]
        masks = np.stack(mask_list)
        masks = np.expand_dims(masks, axis=-1)
        masks = torch.from_numpy(masks).permute(0, 3, 1, 2).contiguous()
        masks = masks.float()
        masks = masks.div(255.0)

        if self.use_cuda:
            masks = masks.cuda()

        return masks


class VideoInpaintIterator(BufferedIterator):
    """
    Video inpainting buffered iterator.
    """
    def __init__(self,
                 **kwargs):
        super(VideoInpaintIterator, self).__init__(**kwargs)

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

        pred_frames = pred_frames * masks + frames * (1 - masks)

        return pred_frames

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


class FrameCollectIterator(BufferedIterator):
    """
    Frame collecting as numpy-array buffered iterator.
    """
    def __init__(self,
                 **kwargs):
        super(FrameCollectIterator, self).__init__(**kwargs)

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

        pred_frames = raw_data_chunk_list[0]

        pred_frames = (((pred_frames + 1.0) / 2.0) * 255).to(torch.uint8)
        pred_frames = pred_frames.permute(0, 2, 3, 1).cpu().detach().numpy()

        return pred_frames

    def _expand_buffer_by(self,
                          data_chunk: Sequence):
        """
        Expand buffer by extra data.

        Parameters
        ----------
        data_chunk : sequence
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
    frame_loader = FrameIterator(
        data=FilePathDirIterator(frames_dir_path),
        image_resize_ratio=image_resize_ratio,
        use_cuda=True)

    mask_loader = MaskIterator(
        mask_dilation=mask_dilation,
        data=FilePathDirIterator(masks_dir_path),
        image_resize_ratio=image_resize_ratio,
        use_cuda=True)

    video_length = len(frame_loader)

    flow_loader = RAFTIterator(
        frames=frame_loader,
        raft_model_path=raft_model_path,
        raft_iters=raft_iters)

    flow_comp_loader = PPRFCIterator(
        flows=flow_loader,
        masks=mask_loader,
        pprfc_model_path=pprfc_model_path)

    image_prop_loader = PPIPIterator(
        frames=frame_loader,
        masks=mask_loader,
        comp_flows=flow_comp_loader)

    image_trans_loader = ProPainterIterator(
        image_prop_loader=image_prop_loader,
        mask_loader=mask_loader,
        flow_comp_loader=flow_comp_loader,
        pp_model_path=pp_model_path)

    video_inpaint_loader = VideoInpaintIterator(
        data=[image_trans_loader, frame_loader, mask_loader])

    frame_collect_loader = FrameCollectIterator(
        data=[video_inpaint_loader])

    vi_step = 10
    video_inpaint_loader_trim_pad = 2
    image_trans_loader_trim_pad = 6
    image_prop_loader_trim_pad = 35
    flow_comp_loader_trim_pad = 3
    flow_loader_trim_pad = 3
    mask_loader_trim_pad = 35
    frame_loader_trim_pad = 2
    for s in range(0, video_length, vi_step):
        e = min(s + vi_step, video_length)
        vi_frames_i = frame_collect_loader[s:e]
        assert (vi_frames_i is not None)
        video_inpaint_loader.trim_buffer_to(max(e - video_inpaint_loader_trim_pad, 0))
        image_trans_loader.trim_buffer_to(max(e - image_trans_loader_trim_pad, 0))
        image_prop_loader.trim_buffer_to(max(e - image_prop_loader_trim_pad, 0))
        flow_comp_loader.trim_buffer_to(max(e - flow_comp_loader_trim_pad, 0))
        flow_loader.trim_buffer_to(max(e - flow_loader_trim_pad, 0))
        mask_loader.trim_buffer_to(max(e - mask_loader_trim_pad, 0))
        frame_loader.trim_buffer_to(max(e - frame_loader_trim_pad, 0))
        torch.cuda.empty_cache()

    assert (frame_collect_loader.start_pos == 0)
    return frame_collect_loader.buffer


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
