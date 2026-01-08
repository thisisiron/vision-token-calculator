import sys

import pytest


TEST_IMAGE_URL = (
    "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"
)


def run_cli(capsys, argv):
    from vt_calculator.cli.main import main

    old_argv = sys.argv
    try:
        sys.argv = ["vt-calc", *argv]
        exit_code = main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    return exit_code, captured.out + captured.err


def test_cli_with_image(capsys, tmp_path):
    from PIL import Image

    img = Image.new("RGB", (512, 512), color=(255, 128, 0))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    exit_code, output = run_cli(capsys, ["--image", str(img_path)])
    assert "VISION TOKEN ANALYSIS REPORT" in output


def test_cli_with_directory_via_image_flag(capsys, tmp_path):
    from PIL import Image

    img1 = Image.new("RGB", (256, 256), color=(255, 0, 0))
    img1.save(tmp_path / "test1.jpg")

    img2 = Image.new("RGB", (512, 512), color=(0, 255, 0))
    img2.save(tmp_path / "test2.jpg")

    exit_code, output = run_cli(capsys, ["--image", str(tmp_path)])
    assert "BATCH ANALYSIS REPORT" in output
    assert "Total Images Processed" in output
    assert "Average Vision Tokens" in output


@pytest.mark.network
def test_cli_with_url(capsys):
    exit_code, output = run_cli(capsys, ["--image", TEST_IMAGE_URL])
    assert "Loading image from URL:" in output
    assert "VISION TOKEN ANALYSIS REPORT" in output


def test_is_url():
    from vt_calculator.utils import is_url

    assert is_url("https://example.com/image.jpg") is True
    assert is_url("http://example.com/image.jpg") is True
    assert is_url("HTTP://EXAMPLE.COM/image.jpg") is True
    assert is_url("/path/to/local/image.jpg") is False
    assert is_url("./relative/path.png") is False
    assert is_url("C:\\Windows\\path.jpg") is False
    assert is_url("") is False
    assert is_url(None) is False


def test_cli_with_size_image(capsys):
    exit_code, output = run_cli(capsys, ["--size", "640", "480"])
    assert "Using dummy image: 640 x 480" in output
    assert "VISION TOKEN ANALYSIS REPORT" in output
    assert "Dummy image" in output


def test_cli_with_size_video(capsys):
    exit_code, output = run_cli(
        capsys, ["--size", "640", "480", "--duration", "2", "--fps", "30"]
    )
    assert "Using dummy video: 480x640" in output
    assert "VIDEO TOKEN ANALYSIS REPORT" in output
    assert "Dummy video" in output


def test_cli_with_video_file(capsys, tmp_path):
    from vt_calculator.utils import create_dummy_video

    video_path = tmp_path / "test.mp4"
    create_dummy_video(str(video_path), width=320, height=240, fps=24, duration=1)

    exit_code, output = run_cli(capsys, ["--video", str(video_path)])
    assert "VIDEO TOKEN ANALYSIS REPORT" in output


def test_cli_compare_with_dummy_image(capsys):
    exit_code, output = run_cli(
        capsys, ["--size", "640", "480", "--compare", "qwen2.5-vl,llava"]
    )
    assert "MODEL COMPARISON" in output
    assert "qwen2.5-vl" in output
    assert "llava" in output
    assert "Best" in output


def test_cli_compare_with_image_file(capsys, tmp_path):
    from PIL import Image

    img = Image.new("RGB", (512, 512), color=(255, 128, 0))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    exit_code, output = run_cli(
        capsys, ["--image", str(img_path), "--compare", "qwen2.5-vl,llava"]
    )
    assert "MODEL COMPARISON" in output
    assert "qwen2.5-vl" in output
    assert "llava" in output


def test_cli_compare_all_models(capsys):
    exit_code, output = run_cli(capsys, ["--size", "256", "256", "--compare", "all"])
    assert "MODEL COMPARISON" in output
    assert "qwen2.5-vl" in output
    assert "internvl3" in output
    assert "Summary" in output


def test_cli_compare_invalid_model(capsys):
    exit_code, output = run_cli(
        capsys, ["--size", "640", "480", "--compare", "qwen2.5-vl,invalid-model-xyz"]
    )
    assert "Unsupported models" in output or "Error" in output


def test_cli_compare_video(capsys):
    exit_code, output = run_cli(
        capsys,
        [
            "--size",
            "640",
            "480",
            "--duration",
            "2",
            "--fps",
            "1",
            "--compare",
            "qwen2.5-vl,llava-next",
        ],
    )
    assert "VIDEO MODEL COMPARISON" in output
    assert "qwen2.5-vl" in output
