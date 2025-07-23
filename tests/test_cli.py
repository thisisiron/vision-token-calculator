import sys
from pathlib import Path


def run_cli(capsys, argv):
    import vt_calculator.calculate as calc

    old_argv = sys.argv
    try:
        sys.argv = ["vt-calc", *argv]
        exit_code = calc.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    return exit_code, captured.out + captured.err


def test_cli_with_size(capsys):
    exit_code, output = run_cli(capsys, ["--size", "640", "480"])
    assert exit_code == 0
    assert "VISION TOKEN ANALYSIS RESULTS" in output
    assert "Image Size (W x H)     : 640 x 480" in output
    assert "Number of Image Tokens" in output


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
