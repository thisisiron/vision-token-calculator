"""
DeepSeek-OCR Vision Token Calculator Tests (TDD)

Token Calculation Formula (from modeling_deepseekocr.py):
- patch_size = 16
- downsample_ratio = 4
- num_row = ceil((image_size // patch_size) / downsample_ratio)

Native Resolution Mode (crop_mode=False):
    Token structure per row: <image> * num_row + <image_newline>
    Total rows: num_row
    End token: <image_seperator>

    Formula: (num_row + 1) * num_row + 1
             ^^^^^^^^       ^^^^^^^^   ^
             row tokens     num rows   end token (<image_seperator>)
             (images+newline)

Gundam Mode (crop_mode=True, base_size=1024, image_size=640):
    global_tokens = (num_row_base + 1) * num_row_base + 1
    local_tokens = (num_row * width_tiles + 1) * (num_row * height_tiles)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^
                   row tokens (images+newline)   total rows
        (only if width_tiles > 1 or height_tiles > 1)
    total = global_tokens + local_tokens

Expected token counts by mode:
- tiny (512):  num_row=8  -> (8+1)*8+1   = 73
- small (640): num_row=10 -> (10+1)*10+1 = 111
- base (1024): num_row=16 -> (16+1)*16+1 = 273
- large (1280): num_row=20 -> (20+1)*20+1 = 421
- gundam: base=273 + dynamic local crops
"""

import math
import pytest


class TestDeepSeekOCRNativeResolution:
    """Native Resolution mode (crop_mode=False) tests.

    In native mode, token count is fixed regardless of input image size.
    The image is resized/padded to the mode's image_size.
    """

    def test_tiny_mode_returns_73_tokens(self):
        """Tiny mode: 512x512 -> num_row=8 -> (8+1)*8+1=73 tokens."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="tiny")
        result = analyst.calculate_image((512, 512))

        assert result["number_of_image_tokens"] == 73
        assert result["mode"] == "tiny"
        assert result["base_size"] == 512

    def test_small_mode_returns_111_tokens(self):
        """Small mode: 640x640 -> num_row=10 -> (10+1)*10+1=111 tokens."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="small")
        result = analyst.calculate_image((640, 640))

        assert result["number_of_image_tokens"] == 111
        assert result["mode"] == "small"
        assert result["base_size"] == 640

    def test_base_mode_returns_273_tokens(self):
        """Base mode: 1024x1024 -> num_row=16 -> (16+1)*16+1=273 tokens."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="base")
        result = analyst.calculate_image((1024, 1024))

        assert result["number_of_image_tokens"] == 273
        assert result["mode"] == "base"
        assert result["base_size"] == 1024

    def test_large_mode_returns_421_tokens(self):
        """Large mode: 1280x1280 -> num_row=20 -> (20+1)*20+1=421 tokens."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="large")
        result = analyst.calculate_image((1280, 1280))

        assert result["number_of_image_tokens"] == 421
        assert result["mode"] == "large"
        assert result["base_size"] == 1280

    def test_native_mode_ignores_input_image_size(self):
        """Native modes produce fixed tokens regardless of input size."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="base")

        # Different input sizes should all produce same token count
        result_small = analyst.calculate_image((100, 100))
        result_medium = analyst.calculate_image((640, 480))
        result_large = analyst.calculate_image((1920, 1080))
        result_huge = analyst.calculate_image((4000, 3000))

        assert result_small["number_of_image_tokens"] == 273
        assert result_medium["number_of_image_tokens"] == 273
        assert result_large["number_of_image_tokens"] == 273
        assert result_huge["number_of_image_tokens"] == 273


class TestDeepSeekOCRGundamMode:
    """Gundam mode (crop_mode=True, base_size=1024, image_size=640) tests.

    In Gundam mode:
    - Global view always uses base_size=1024 -> 273 tokens
    - If image > 640x640, dynamic crops are added
    - Local crop tokens = (10 * width_tiles + 1) * (10 * height_tiles)
    """

    def test_gundam_small_image_no_crops(self):
        """Image <= 640x640: no crops, only global view -> 273 tokens."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")

        # Images at or below 640x640 threshold
        result_exact = analyst.calculate_image((640, 640))
        result_small = analyst.calculate_image((500, 400))

        # Only global tokens (no local crops)
        assert result_exact["number_of_image_tokens"] == 273
        assert result_small["number_of_image_tokens"] == 273
        assert result_exact.get("tile_grid") == (1, 1)
        assert result_exact.get("num_local_tokens", 0) == 0

    def test_gundam_wide_image_4x2_crops(self):
        """Wide image (1920x1080, aspect~1.78) -> (2,4) tiles (H×W).

        Aspect ratio 1.78 is equidistant from (2,1)=2.0 and (4,2)=2.0.
        Tie-breaker favors more tiles when area justifies it.

        - Global: 273 tokens
        - Local: (10*4+1)*(10*2) = 41*20 = 820 tokens
        - Total: 273 + 820 = 1093 tokens
        """
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((1080, 1920))  # (height, width)

        expected_local = (10 * 4 + 1) * (10 * 2)  # 820
        expected_total = 273 + expected_local  # 1093

        assert result["tile_grid"] == (2, 4)  # (height_tiles, width_tiles) = (H×W)
        assert result["number_of_image_tokens"] == expected_total

    def test_gundam_tall_image_2x4_crops(self):
        """Tall image (1080x1920, aspect~0.56) -> (4,2) tiles (H×W).

        Aspect ratio 0.56 is equidistant from (1,2)=0.5 and (2,4)=0.5.
        Tie-breaker favors more tiles when area justifies it.

        - Global: 273 tokens
        - Local: (10*2+1)*(10*4) = 21*40 = 840 tokens
        - Total: 273 + 840 = 1113 tokens
        """
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((1920, 1080))  # (height, width)

        expected_local = (10 * 2 + 1) * (10 * 4)  # 840
        expected_total = 273 + expected_local  # 1113

        assert result["tile_grid"] == (4, 2)  # (height_tiles, width_tiles) = (H×W)
        assert result["number_of_image_tokens"] == expected_total

    def test_gundam_square_large_image_2x2_crops(self):
        """Square large image (1280x1280) -> (2,2) tiles (min_num=2).

        - Global: 273 tokens
        - Local: (10*2+1)*(10*2) = 21*20 = 420 tokens
        - Total: 273 + 420 = 693 tokens
        """
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((1280, 1280))

        expected_local = (10 * 2 + 1) * (10 * 2)  # 420
        expected_total = 273 + expected_local  # 693

        assert result["tile_grid"] == (2, 2)
        assert result["number_of_image_tokens"] == expected_total

    def test_gundam_very_wide_image_4x1_crops(self):
        """Very wide image (2560x720, aspect~3.56) -> (1,4) tiles (H×W).

        Aspect ratio 3.56 is closer to (4,1)=4.0 than (3,1)=3.0.
        diff(3.56, 4.0) = 0.44 < diff(3.56, 3.0) = 0.56

        - Global: 273 tokens
        - Local: (10*4+1)*(10*1) = 41*10 = 410 tokens
        - Total: 273 + 410 = 683 tokens
        """
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((720, 2560))  # (height, width)

        expected_local = (10 * 4 + 1) * (10 * 1)  # 410
        expected_total = 273 + expected_local  # 683

        assert result["tile_grid"] == (1, 4)  # (height_tiles, width_tiles) = (H×W)
        assert result["number_of_image_tokens"] == expected_total


class TestDeepSeekOCRTileCalculation:
    """Dynamic tile (crop) calculation utility tests.

    Tile selection logic (from get_optimal_tiled_canvas):
    - Generate all (i,j) pairs where min_num <= i*j <= max_num
    - Select the pair with aspect ratio closest to image aspect ratio
    - On tie, prefer more tiles if area > 0.5 * tile_area
    """

    def test_count_tiles_wide_image(self):
        """Wide image should select tiles with width > height."""
        from vt_calculator.analysts.tools import count_tiles_deepseek

        # 1920/1080 = 1.78, but tie-breaker prefers more tiles
        result = count_tiles_deepseek(1920, 1080, min_num=2, max_num=9, image_size=640)
        assert result[0] > result[1]  # width_tiles > height_tiles
        assert result == (4, 2)  # (2,1) and (4,2) have same aspect, more tiles wins

    def test_count_tiles_tall_image(self):
        """Tall image should select tiles with height > width."""
        from vt_calculator.analysts.tools import count_tiles_deepseek

        # 1080/1920 = 0.56, but tie-breaker prefers more tiles
        result = count_tiles_deepseek(1080, 1920, min_num=2, max_num=9, image_size=640)
        assert result[1] > result[0]  # height_tiles > width_tiles
        assert result == (2, 4)  # (1,2) and (2,4) have same aspect, more tiles wins

    def test_count_tiles_square_image(self):
        """Square image should select square tiles."""
        from vt_calculator.analysts.tools import count_tiles_deepseek

        result = count_tiles_deepseek(1280, 1280, min_num=2, max_num=9, image_size=640)
        assert result[0] == result[1]  # square tiles
        assert result == (2, 2)  # min_num=2, so smallest square is 2x2

    def test_count_tiles_very_wide_image(self):
        """Very wide image (aspect > 3) should select (4,1) tiles."""
        from vt_calculator.analysts.tools import count_tiles_deepseek

        # 2560/720 = 3.56, closer to 4.0 than 3.0
        result = count_tiles_deepseek(2560, 720, min_num=2, max_num=9, image_size=640)
        assert result == (4, 1)  # aspect 4.0 is closest to 3.56

    def test_count_tiles_below_threshold(self):
        """Images at or below 640x640 return (1,1) - no crops."""
        from vt_calculator.analysts.tools import count_tiles_deepseek

        result_exact = count_tiles_deepseek(640, 640, min_num=2, max_num=9, image_size=640)
        result_small = count_tiles_deepseek(500, 400, min_num=2, max_num=9, image_size=640)

        assert result_exact == (1, 1)
        assert result_small == (1, 1)

    def test_count_tiles_above_threshold(self):
        """Images above 640x640 get dynamic tiling."""
        from vt_calculator.analysts.tools import count_tiles_deepseek

        # 1920/1080 = 1.78, (4,2) aspect = 2.0
        result = count_tiles_deepseek(1920, 1080, min_num=2, max_num=9, image_size=640)
        assert result == (4, 2)


class TestDeepSeekOCROutputFormat:
    """Test calculate_image return value structure."""

    def test_output_has_required_keys(self):
        """Return dict must have all required keys."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="base")
        result = analyst.calculate_image((1024, 1024))

        required_keys = [
            "processing_method",
            "image_token",
            "image_newline_token",
            "image_separator_token",
            "number_of_image_tokens",
            "image_token_format",
            "image_size",
            "resized_size",
            "patch_size",
            "patch_grid",
            "total_patches",
            "has_global_patch",
            "mode",
            "base_size",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_image_token_is_tuple(self):
        """image_token must be (token_name, count) tuple with pure <image> count."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="base")
        result = analyst.calculate_image((1024, 1024))

        assert isinstance(result["image_token"], tuple)
        assert len(result["image_token"]) == 2
        assert result["image_token"][0] == "<image>"
        assert isinstance(result["image_token"][1], int)
        # Base mode: 16*16 = 256 pure <image> tokens
        assert result["image_token"][1] == 256

    def test_gundam_mode_has_crop_info(self):
        """Gundam mode should include crop information."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((1920, 1080))

        assert result["processing_method"] == "tile_based"
        assert "tile_grid" in result
        assert "number_of_tiles" in result
        assert "num_global_tokens" in result
        assert "num_local_tokens" in result
        assert isinstance(result["tile_grid"], tuple)


class TestDeepSeekOCRVideo:
    """Video support tests - DeepSeek-OCR does not support video."""

    def test_calculate_video_raises_not_implemented(self):
        """calculate_video should raise NotImplementedError."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="base")

        with pytest.raises(NotImplementedError) as exc_info:
            analyst.calculate_video(
                {"width": 1920, "height": 1080, "duration": 10.0}
            )

        assert "video" in str(exc_info.value).lower()


class TestDeepSeekOCRRegistration:
    """Model registration and factory tests."""

    def test_all_modes_in_supported_models(self):
        """All 5 modes should be registered as supported models."""
        from vt_calculator.analysts import SUPPORTED_MODELS

        expected_models = [
            "deepseek-ocr-tiny",
            "deepseek-ocr-small",
            "deepseek-ocr-base",
            "deepseek-ocr-large",
            "deepseek-ocr-gundam",
        ]
        for model in expected_models:
            assert model in SUPPORTED_MODELS, f"Model {model} not in SUPPORTED_MODELS"

    def test_model_to_hf_id_is_none(self):
        """DeepSeek-OCR models should have None as HF ID (no AutoProcessor)."""
        from vt_calculator.analysts import MODEL_TO_HF_ID

        for mode in ["tiny", "small", "base", "large", "gundam"]:
            model_name = f"deepseek-ocr-{mode}"
            assert model_name in MODEL_TO_HF_ID
            assert MODEL_TO_HF_ID[model_name] is None

    def test_load_analyst_returns_correct_class(self):
        """load_analyst should return DeepSeekOCRAnalyst with correct mode."""
        from vt_calculator.analysts import load_analyst
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = load_analyst("deepseek-ocr-base")

        assert isinstance(analyst, DeepSeekOCRAnalyst)
        assert analyst.mode == "base"

    def test_load_analyst_all_modes(self):
        """load_analyst should work for all 5 modes."""
        from vt_calculator.analysts import load_analyst
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        modes = ["tiny", "small", "base", "large", "gundam"]
        for mode in modes:
            analyst = load_analyst(f"deepseek-ocr-{mode}")
            assert isinstance(analyst, DeepSeekOCRAnalyst)
            assert analyst.mode == mode


class TestDeepSeekOCREdgeCases:
    """Edge case and error handling tests."""

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode should raise ValueError."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        with pytest.raises(ValueError) as exc_info:
            DeepSeekOCRAnalyst(mode="invalid_mode")

        assert "invalid_mode" in str(exc_info.value).lower()

    def test_very_small_image_in_gundam_mode(self):
        """Very small images should still work in Gundam mode."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((50, 50))

        # Small images get no crops, only global view
        assert result["number_of_image_tokens"] == 273
        assert result["tile_grid"] == (1, 1)

    def test_very_large_image_respects_max_crops(self):
        """Very large images should respect max_crops=9 limit."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((5000, 5000))

        # Should be limited to 3x3 = 9 crops max
        height_tiles, width_tiles = result["tile_grid"]  # (H×W)
        assert height_tiles * width_tiles <= 9

    def test_extreme_aspect_ratio_wide(self):
        """Extremely wide images should work."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((480, 6000))  # very wide

        assert result["number_of_image_tokens"] > 273  # has crops
        height_tiles, width_tiles = result["tile_grid"]  # (H×W)
        assert width_tiles > height_tiles

    def test_extreme_aspect_ratio_tall(self):
        """Extremely tall images should work."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((6000, 480))  # very tall

        assert result["number_of_image_tokens"] > 273  # has crops
        height_tiles, width_tiles = result["tile_grid"]  # (H×W)
        assert height_tiles > width_tiles


class TestDeepSeekOCRConstants:
    """Test that analyst uses correct constants from DeepSeek-OCR."""

    def test_patch_size_is_16(self):
        """Patch size should be 16."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="base")
        assert analyst.patch_size == 16

    def test_downsample_ratio_is_4(self):
        """Downsample ratio should be 4."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="base")
        assert analyst.downsample_ratio == 4

    def test_image_token_is_correct(self):
        """Image token should be '<image>'."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="base")
        assert analyst.image_token == "<image>"


class TestDeepSeekOCRTokenFormula:
    """Verify token calculation formula matches original implementation."""

    @pytest.mark.parametrize(
        "image_size,expected_tokens",
        [
            (512, 73),    # ceil(32/4)=8 -> (8+1)*8+1=73
            (640, 111),   # ceil(40/4)=10 -> (10+1)*10+1=111
            (1024, 273),  # ceil(64/4)=16 -> (16+1)*16+1=273
            (1280, 421),  # ceil(80/4)=20 -> (20+1)*20+1=421
        ],
    )
    def test_native_token_formula(self, image_size, expected_tokens):
        """Verify: tokens = (num_row + 1) * num_row + 1."""
        patch_size = 16
        downsample_ratio = 4

        num_row = math.ceil((image_size // patch_size) / downsample_ratio)
        calculated_tokens = (num_row + 1) * num_row + 1

        assert calculated_tokens == expected_tokens

    @pytest.mark.parametrize(
        "width_tiles,height_tiles,expected_local",
        [
            (2, 1, 210),  # (10*2+1)*(10*1) = 21*10 = 210
            (1, 2, 220),  # (10*1+1)*(10*2) = 11*20 = 220
            (2, 2, 420),  # (10*2+1)*(10*2) = 21*20 = 420
            (3, 1, 310),  # (10*3+1)*(10*1) = 31*10 = 310
            (1, 3, 330),  # (10*1+1)*(10*3) = 11*30 = 330
            (3, 2, 620),  # (10*3+1)*(10*2) = 31*20 = 620
            (3, 3, 930),  # (10*3+1)*(10*3) = 31*30 = 930
        ],
    )
    def test_gundam_local_token_formula(self, width_tiles, height_tiles, expected_local):
        """Verify: local_tokens = (num_row * w + 1) * (num_row * h)."""
        num_row = 10  # for image_size=640

        calculated_local = (num_row * width_tiles + 1) * (num_row * height_tiles)
        assert calculated_local == expected_local


class TestDeepSeekOCRTokenSeparation:
    """Test that token types are properly separated."""

    def test_native_mode_separates_image_tokens(self):
        """Native mode should separate <image>, <image_newline>, <image_separator>."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="tiny")
        result = analyst.calculate_image((512, 512))

        # num_row = 8 for tiny mode
        # Pure <image> tokens: 8 * 8 = 64
        assert result["image_token"][1] == 64
        # <image_newline> tokens: 8 (one per row)
        assert result["image_newline_token"][1] == 8
        # <image_separator> token: 1
        assert result["image_separator_token"][1] == 1
        # Total: 64 + 8 + 1 = 73
        assert result["number_of_image_tokens"] == 73

    def test_gundam_mode_separates_image_tokens_no_crops(self):
        """Gundam mode without crops should separate tokens like native mode."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((640, 640))

        # num_row_base = 16 for base_size=1024
        # Pure <image> tokens: 16 * 16 = 256
        assert result["image_token"][1] == 256
        # <image_newline> tokens: 16
        assert result["image_newline_token"][1] == 16
        # <image_separator> token: 1
        assert result["image_separator_token"][1] == 1
        # Total: 256 + 16 + 1 = 273
        assert result["number_of_image_tokens"] == 273

    def test_gundam_mode_separates_image_tokens_with_crops(self):
        """Gundam mode with crops should separate global and local tokens."""
        from vt_calculator.analysts.analyst import DeepSeekOCRAnalyst

        analyst = DeepSeekOCRAnalyst(mode="gundam")
        result = analyst.calculate_image((1280, 1280))  # (2,2) tiles

        # Global: num_row=16 -> 256 <image> + 16 <newline> + 1 <sep> = 273
        # Local: (10*2+1)*(10*2) = 420 tokens
        #   - <image> tokens: 10*2 * 10*2 = 400
        #   - <newline> tokens: 10*2 = 20
        # Total tokens: 273 + 420 = 693
        assert result["number_of_image_tokens"] == 693
        # Pure <image>: 256 (global) + 400 (local) = 656
        assert result["image_token"][1] == 656
        # <image_newline>: 16 (global) + 20 (local) = 36
        assert result["image_newline_token"][1] == 36
        # <image_separator>: 1 (only in global)
        assert result["image_separator_token"][1] == 1
