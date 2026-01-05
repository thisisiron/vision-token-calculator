import sys
from pathlib import Path

import pytest


TEST_IMAGE_URL = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"


def run_cli(capsys, argv):
    import vt_calculator.core.calculator as calc

    old_argv = sys.argv
    try:
        sys.argv = ["vt-calc", *argv]
        exit_code = calc.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    return exit_code, captured.out + captured.err


def test_cli_with_image(capsys):
    # Use a small bundled image if present, otherwise fall back to the test directory one
    repo_root = Path(__file__).resolve().parents[1]
    default_image = repo_root / "test_image.jpg"
    if not default_image.exists():
        default_image = repo_root / "test_images" / "test_6_512x512.jpg"

    exit_code, output = run_cli(capsys, ["--image", str(default_image)])
    assert exit_code == 0
    assert "VISION TOKEN ANALYSIS RESULTS" in output
    assert "Existing image:" in output
    assert "Number of Image Tokens" in output


def test_cli_with_directory_via_image_flag(capsys, tmp_path):
    # Copy a couple of images into a temp directory to ensure isolation
    repo_root = Path(__file__).resolve().parents[1]
    img1 = repo_root / "test_images" / "test_7_256x256.jpg"
    img2 = repo_root / "test_images" / "test_6_512x512.jpg"
    dst1 = tmp_path / img1.name
    dst2 = tmp_path / img2.name
    dst1.write_bytes(img1.read_bytes())
    dst2.write_bytes(img2.read_bytes())

    exit_code, output = run_cli(capsys, ["--image", str(tmp_path)])
    assert exit_code == 0
    assert "BATCH ANALYSIS RESULTS" in output
    assert "Total Images Processed" in output
    assert "Average Vision Tokens" in output


@pytest.mark.network
def test_cli_with_url(capsys):
    exit_code, output = run_cli(capsys, ["--image", TEST_IMAGE_URL])
    assert exit_code is None or exit_code == 0
    assert "Loading image from URL:" in output
    assert "VISION TOKEN ANALYSIS" in output


@pytest.mark.network
def test_count_image_tokens_with_url():
    from vt_calculator import count_image_tokens

    result = count_image_tokens(TEST_IMAGE_URL)
    assert "image_size" in result
    assert "number_of_image_tokens" in result or "image_token" in result


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
