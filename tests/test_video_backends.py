import pytest

from vt_calculator.video import (
    VideoBackend,
    get_default_backend,
    get_video_metadata,
    extract_video_frames,
)
from vt_calculator.video.backends import (
    is_torchcodec_available,
    is_decord_available,
    is_torchvision_available,
    is_pyav_available,
)


class TestBackendAvailability:
    def test_pyav_is_available(self):
        assert is_pyav_available() is True

    def test_backend_enum_values(self):
        assert VideoBackend.TORCHCODEC.value == "torchcodec"
        assert VideoBackend.DECORD.value == "decord"
        assert VideoBackend.TORCHVISION.value == "torchvision"
        assert VideoBackend.PYAV.value == "pyav"


class TestBackendSelection:
    def test_default_backend_is_valid(self):
        backend = get_default_backend()
        assert isinstance(backend, VideoBackend)
        assert backend in [
            VideoBackend.TORCHCODEC,
            VideoBackend.DECORD,
            VideoBackend.TORCHVISION,
            VideoBackend.PYAV,
        ]

    def test_pyav_is_always_fallback(self):
        backend = get_default_backend()
        assert backend is not None

    @pytest.mark.skipif(
        not is_torchcodec_available(), reason="torchcodec not available"
    )
    def test_torchcodec_priority(self):
        backend = get_default_backend()
        assert backend == VideoBackend.TORCHCODEC

    @pytest.mark.skipif(
        is_torchcodec_available() or not is_decord_available(),
        reason="decord not highest priority",
    )
    def test_decord_priority(self):
        backend = get_default_backend()
        assert backend == VideoBackend.DECORD


class TestVideoMetadata:
    @pytest.fixture
    def video_file(self, dummy_video):
        return dummy_video

    def test_get_metadata_pyav(self, video_file):
        metadata = get_video_metadata(video_file, backend=VideoBackend.PYAV)

        assert "width" in metadata
        assert "height" in metadata
        assert "fps" in metadata
        assert "duration" in metadata
        assert "total_frames" in metadata

        assert metadata["width"] == 336
        assert metadata["height"] == 336
        assert metadata["fps"] == 3.0
        assert metadata["duration"] > 0

    def test_get_metadata_auto_backend(self, video_file):
        metadata = get_video_metadata(video_file)

        assert "width" in metadata
        assert "height" in metadata
        assert "fps" in metadata
        assert "duration" in metadata
        assert "total_frames" in metadata

    @pytest.mark.skipif(not is_decord_available(), reason="decord not installed")
    def test_get_metadata_decord(self, video_file):
        metadata = get_video_metadata(video_file, backend=VideoBackend.DECORD)

        assert metadata["width"] == 336
        assert metadata["height"] == 336
        assert metadata["fps"] == pytest.approx(3.0, rel=0.1)

    @pytest.mark.skipif(
        not is_torchvision_available(), reason="torchvision not installed"
    )
    def test_get_metadata_torchvision(self, video_file):
        metadata = get_video_metadata(video_file, backend=VideoBackend.TORCHVISION)

        assert metadata["width"] == 336
        assert metadata["height"] == 336

    @pytest.mark.skipif(
        not is_torchcodec_available(), reason="torchcodec not installed"
    )
    def test_get_metadata_torchcodec(self, video_file):
        metadata = get_video_metadata(video_file, backend=VideoBackend.TORCHCODEC)

        assert metadata["width"] == 336
        assert metadata["height"] == 336
        assert metadata["fps"] == pytest.approx(3.0, rel=0.1)


class TestEnvironmentVariableOverride:
    def test_force_backend_env_var(self, monkeypatch, dummy_video):
        monkeypatch.setenv("FORCE_VIDEO_BACKEND", "PYAV")

        get_default_backend.cache_clear()

        backend = get_default_backend()
        assert backend == VideoBackend.PYAV

    def test_invalid_force_backend_fallsback(self, monkeypatch, dummy_video):
        monkeypatch.setenv("FORCE_VIDEO_BACKEND", "INVALID")

        get_default_backend.cache_clear()

        backend = get_default_backend()
        assert backend in [
            VideoBackend.TORCHCODEC,
            VideoBackend.DECORD,
            VideoBackend.TORCHVISION,
            VideoBackend.PYAV,
        ]


class TestFrameExtraction:
    @pytest.fixture
    def video_file(self, dummy_video):
        return dummy_video

    def test_extract_frames_pyav(self, video_file):
        frames_result = extract_video_frames(
            video_file, fps=1.0, backend=VideoBackend.PYAV
        )

        assert frames_result.frames is not None
        assert isinstance(frames_result.frames, pytest.importorskip("torch").Tensor)
        assert frames_result.frames.ndim == 4
        assert frames_result.sample_fps > 0

    def test_extract_frames_auto_backend(self, video_file):
        frames_result = extract_video_frames(video_file, fps=1.0)

        assert frames_result.frames is not None
        assert isinstance(frames_result.frames, pytest.importorskip("torch").Tensor)
        assert frames_result.frames.ndim == 4

    @pytest.mark.skipif(not is_decord_available(), reason="decord not installed")
    def test_extract_frames_decord(self, video_file):
        frames_result = extract_video_frames(
            video_file, fps=1.0, backend=VideoBackend.DECORD
        )

        assert frames_result.frames is not None
        assert isinstance(frames_result.frames, pytest.importorskip("torch").Tensor)
        assert frames_result.frames.ndim == 4

    @pytest.mark.skipif(
        not is_torchvision_available(), reason="torchvision not installed"
    )
    def test_extract_frames_torchvision(self, video_file):
        frames_result = extract_video_frames(
            video_file, fps=1.0, backend=VideoBackend.TORCHVISION
        )

        assert frames_result.frames is not None
        assert isinstance(frames_result.frames, pytest.importorskip("torch").Tensor)
        assert frames_result.frames.ndim == 4

    @pytest.mark.skipif(
        not is_torchcodec_available(), reason="torchcodec not installed"
    )
    def test_extract_frames_torchcodec(self, video_file):
        frames_result = extract_video_frames(
            video_file, fps=1.0, backend=VideoBackend.TORCHCODEC
        )

        assert frames_result.frames is not None
        assert isinstance(frames_result.frames, pytest.importorskip("torch").Tensor)
        assert frames_result.frames.ndim == 4


class TestErrorHandling:
    def test_nonexistent_video_raises_error(self):
        with pytest.raises(Exception):
            get_video_metadata("/nonexistent/video.mp4")
