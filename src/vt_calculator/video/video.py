"""Unified video processing module with multiple backend support."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

from .backends import VideoBackend, get_default_backend

logger = logging.getLogger(__name__)

TORCHCODEC_NUM_THREADS = 8
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int


@dataclass
class VideoFrames:
    frames: torch.Tensor
    sample_fps: float


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_nframes(
    total_frames: int,
    video_fps: float,
    target_fps: Optional[float] = None,
    min_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> int:
    if target_fps is None:
        target_fps = FPS

    if min_frames is None:
        min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)

    if max_frames is None:
        max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)

    nframes = total_frames / video_fps * target_fps
    if nframes > total_frames:
        logger.warning(
            f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
        )
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(int(nframes), FRAME_FACTOR)

    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(
            f"nframes should be in interval [{FRAME_FACTOR}, {total_frames}], "
            f"but got {nframes}."
        )

    return int(nframes)


def calculate_video_frame_range(
    video_metadata: VideoMetadata,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Tuple[int, int, int]:
    total_frames = video_metadata.total_frames
    video_fps = video_metadata.fps

    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    if start_time is None and end_time is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    video_start_clamped = 0.0
    video_end_clamped = max_duration

    if start_time is not None:
        video_start_clamped = max(0.0, min(start_time, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0

    if end_time is not None:
        video_end_clamped = max(0.0, min(end_time, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} "
            f"(at {video_start_clamped}s) "
            f"exceeds end frame {end_frame} "
            f"(at {video_end_clamped}s). "
            f"Video duration: {max_duration:.2f}s "
            f"({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(
        f"calculate_video_frame_range: start_frame={start_frame}, "
        f"end_frame={end_frame}, total_frames={total_frames} "
        f"from start_time={start_time}, end_time={end_time}, fps={video_fps:.3f}"
    )

    return start_frame, end_frame, end_frame - start_frame + 1


class VideoReader(ABC):
    @abstractmethod
    def get_metadata(self, video_path: str) -> VideoMetadata:
        pass

    @abstractmethod
    def extract_frames(
        self,
        video_path: str,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> torch.Tensor:
        pass


class TorchCodecReader(VideoReader):
    def get_metadata(self, video_path: str) -> VideoMetadata:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
        metadata = decoder.metadata

        return VideoMetadata(
            width=metadata.width,
            height=metadata.height,
            fps=metadata.average_fps,
            duration=metadata.duration_seconds,
            total_frames=metadata.num_frames,
        )

    def extract_frames(
        self,
        video_path: str,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> torch.Tensor:
        from torchcodec.decoders import VideoDecoder

        metadata = self.get_metadata(video_path)
        start_frame, end_frame, frame_count = calculate_video_frame_range(
            metadata, start_time, end_time
        )
        nframes = smart_nframes(frame_count, metadata.fps, fps, max_frames=max_frames)

        decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)

        idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
        frames = decoder.get_frames_at(indices=idx).data

        return frames


class DecordReader(VideoReader):
    def get_metadata(self, video_path: str) -> VideoMetadata:
        import decord

        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)

        return VideoMetadata(
            width=vr[0].shape[1],
            height=vr[0].shape[0],
            fps=fps,
            duration=total_frames / fps if fps > 0 else 0.0,
            total_frames=total_frames,
        )

    def extract_frames(
        self,
        video_path: str,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> torch.Tensor:
        import decord

        metadata = self.get_metadata(video_path)
        start_frame, end_frame, frame_count = calculate_video_frame_range(
            metadata, start_time, end_time
        )
        nframes = smart_nframes(frame_count, metadata.fps, fps, max_frames=max_frames)

        vr = decord.VideoReader(video_path)

        idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
        frames_np = vr.get_batch(idx).asnumpy()
        frames = torch.tensor(frames_np).permute(0, 3, 1, 2)

        return frames


class TorchvisionReader(VideoReader):
    def get_metadata(self, video_path: str) -> VideoMetadata:
        from torchvision.io import read_video

        video, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
        fps = info["video_fps"]
        total_frames = video.shape[0]

        return VideoMetadata(
            width=video.shape[3],
            height=video.shape[2],
            fps=fps,
            duration=total_frames / fps if fps > 0 else 0.0,
            total_frames=total_frames,
        )

    def extract_frames(
        self,
        video_path: str,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> torch.Tensor:
        from torchvision.io import read_video

        metadata = self.get_metadata(video_path)
        start_frame, end_frame, frame_count = calculate_video_frame_range(
            metadata, start_time, end_time
        )
        nframes = smart_nframes(frame_count, metadata.fps, fps, max_frames=max_frames)

        video, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")

        idx = torch.linspace(start_frame, end_frame, nframes).round().long()
        frames = video[idx]

        return frames


class PyAVReader(VideoReader):
    def get_metadata(self, video_path: str) -> VideoMetadata:
        import av

        container = av.open(video_path)
        video_stream = container.streams.video[0]

        width = video_stream.width
        height = video_stream.height
        fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0
        duration = (
            float(container.duration / av.time_base) if container.duration else 0.0
        )
        total_frames = video_stream.frames

        container.close()

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            total_frames=total_frames,
        )

    def extract_frames(
        self,
        video_path: str,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> torch.Tensor:
        import av
        import numpy as np

        metadata = self.get_metadata(video_path)
        start_frame, end_frame, frame_count = calculate_video_frame_range(
            metadata, start_time, end_time
        )
        nframes = smart_nframes(frame_count, metadata.fps, fps, max_frames=max_frames)

        container = av.open(video_path)

        idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
        indices_set = set(idx)
        frames_list = []
        frame_idx = 0

        for frame in container.decode(video=0):
            if frame_idx in indices_set:
                frame_np = frame.to_ndarray(format="rgb24")
                frames_list.append(frame_np)
            frame_idx += 1
            if frame_idx > max(idx):
                break

        container.close()

        frames_array = np.stack(frames_list, axis=0)
        frames = torch.from_numpy(frames_array).permute(0, 3, 1, 2)

        return frames


def get_reader(backend: Optional[VideoBackend] = None) -> VideoReader:
    if backend is None:
        backend = get_default_backend()

    readers = {
        VideoBackend.TORCHCODEC: TorchCodecReader,
        VideoBackend.DECORD: DecordReader,
        VideoBackend.TORCHVISION: TorchvisionReader,
        VideoBackend.PYAV: PyAVReader,
    }

    reader_class = readers.get(backend)
    if reader_class is None:
        raise ValueError(f"Unknown backend: {backend}")

    return reader_class()


def get_video_metadata(
    video_path: str, backend: Optional[VideoBackend] = None
) -> dict[str, int | float]:
    reader = get_reader(backend)
    metadata = reader.get_metadata(video_path)

    return {
        "width": metadata.width,
        "height": metadata.height,
        "fps": metadata.fps,
        "duration": metadata.duration,
        "total_frames": metadata.total_frames,
    }


def extract_video_frames(
    video_path: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    backend: Optional[VideoBackend] = None,
) -> VideoFrames:
    reader = get_reader(backend)
    metadata = reader.get_metadata(video_path)

    frames = reader.extract_frames(
        video_path,
        fps=fps,
        max_frames=max_frames,
        start_time=start_time,
        end_time=end_time,
    )

    start_frame, end_frame, frame_count = calculate_video_frame_range(
        metadata, start_time, end_time
    )
    nframes = smart_nframes(
        total_frames=frame_count,
        video_fps=metadata.fps,
        target_fps=fps,
        max_frames=max_frames,
    )

    sample_fps = nframes / metadata.duration if metadata.duration > 0 else 0.0

    return VideoFrames(frames=frames, sample_fps=sample_fps)
