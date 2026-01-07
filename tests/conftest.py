import os
import pytest
import tempfile

from vt_calculator.utils import create_dummy_video


@pytest.fixture
def dummy_video():
    """Create a dummy MP4 video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name

    create_dummy_video(file_path=video_path, width=336, height=336, fps=3, duration=10)

    yield video_path

    if os.path.exists(video_path):
        os.remove(video_path)
