"""
VLM Test Template - Copy this file to create tests for a new VLM.

Usage:
1. Copy this file to `test_<model_name>.py`
2. Replace all occurrences of:
   - `XxxModel` with your model name (e.g., `Phi4Multimodal`)
   - `xxx-model` with your model CLI name (e.g., `phi4-multimodal`)
   - `XxxModelAnalyst` with your analyst class name
3. Update the docstring with model-specific formula documentation
4. Implement the test cases with model-specific values

Required Test Classes:
- TestXxxModelBasic: Basic token calculation and constants
- TestXxxModelRegistration: Model registration and loading
- TestXxxModelProcessingModes: Processing modes (various image sizes/ratios)
- TestXxxModelOutputFormat: Output dictionary required keys
- TestXxxModelTokenFormula: Formula verification with parametrize
"""

import pytest


# =============================================================================
# Configuration - Update these for your model
# =============================================================================

MODEL_NAME = "xxx-model"  # CLI model name (e.g., "phi4-multimodal")
ANALYST_CLASS_NAME = "XxxModelAnalyst"  # Class name in analyst.py

# Expected token counts for basic tests (update with your model's values)
BASIC_TEST_CASES = [
    # (image_size, expected_tokens, description)
    ((448, 448), 273, "Small image - no tiling"),
    ((896, 896), 1329, "Medium image - 2x2 tiles"),
]

# Required output keys for different processing methods
REQUIRED_KEYS_FIXED = [
    "processing_method",
    "image_token",
    "image_size",
    "resized_size",
    "patch_size",
]

REQUIRED_KEYS_TILED = REQUIRED_KEYS_FIXED + [
    "tile_grid",
    "number_of_tiles",
]

# Token formula test cases for parametrize
# (input_params, expected_tokens)
TOKEN_FORMULA_CASES = [
    ((1, 1), 273),
    ((2, 2), 1329),
]


# =============================================================================
# Test Classes
# =============================================================================


class TestXxxModelBasic:
    """Basic token calculation tests.

    Verifies:
    - Token counts for standard image sizes match expected values
    - Model constants (patch_size, etc.) are correct
    """

    @pytest.mark.parametrize("image_size,expected_tokens,description", BASIC_TEST_CASES)
    def test_basic_token_count(self, image_size, expected_tokens, description):
        """Verify basic token count for standard image sizes."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image(image_size)

        # Check token count - adapt key name to your model's output
        actual_tokens = result.get("number_of_vision_tokens") or result.get("image_token", [None, 0])[1]
        assert actual_tokens == expected_tokens, f"{description}: expected {expected_tokens}, got {actual_tokens}"

    def test_analyst_has_required_attributes(self):
        """Verify analyst has required attributes."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)

        # Common required attributes - adjust as needed
        assert hasattr(analyst, "patch_size")
        assert hasattr(analyst, "image_token")


class TestXxxModelRegistration:
    """Model registration and loading tests.

    Verifies:
    - Model is in SUPPORTED_MODELS
    - load_analyst returns correct class
    """

    def test_model_in_supported_models(self):
        """Model should be registered in SUPPORTED_MODELS."""
        from vt_calculator.analysts import SUPPORTED_MODELS

        assert MODEL_NAME in SUPPORTED_MODELS, f"{MODEL_NAME} not in SUPPORTED_MODELS"

    def test_load_analyst_returns_correct_class(self):
        """load_analyst should return the correct analyst class."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        assert analyst.__class__.__name__ == ANALYST_CLASS_NAME


class TestXxxModelProcessingModes:
    """Processing mode tests for various image sizes and aspect ratios.

    Verifies:
    - Wide images get appropriate tile configuration
    - Tall images get appropriate tile configuration
    - Very large images respect max tile limits
    - Very small images are handled correctly
    """

    def test_wide_image_processing(self):
        """Wide image (1920x1080) should get width > height tiles."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image((1080, 1920))  # (height, width)

        # Verify wide image gets more width tiles
        if "tile_grid" in result:
            h_tiles, w_tiles = result["tile_grid"]
            assert w_tiles >= h_tiles, "Wide image should have width_tiles >= height_tiles"

    def test_tall_image_processing(self):
        """Tall image (1080x1920) should get height > width tiles."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image((1920, 1080))  # (height, width)

        # Verify tall image gets more height tiles
        if "tile_grid" in result:
            h_tiles, w_tiles = result["tile_grid"]
            assert h_tiles >= w_tiles, "Tall image should have height_tiles >= width_tiles"

    def test_very_large_image_respects_max_tiles(self):
        """Very large images should not exceed maximum tile count."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image((5000, 5000))

        # Verify tile count is bounded (adjust max as needed)
        if "tile_grid" in result:
            h_tiles, w_tiles = result["tile_grid"]
            assert h_tiles * w_tiles <= 36, "Tile count should be bounded"

    def test_very_small_image_no_error(self):
        """Very small images should not cause errors."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        # Should not raise exception
        result = analyst.calculate_image((32, 32))
        assert result is not None

    def test_extreme_aspect_ratio_wide(self):
        """Extremely wide images (10:1) should not cause errors."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image((100, 1000))  # 10:1 aspect ratio
        assert result is not None

    def test_extreme_aspect_ratio_tall(self):
        """Extremely tall images (1:10) should not cause errors."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image((1000, 100))  # 1:10 aspect ratio
        assert result is not None


class TestXxxModelOutputFormat:
    """Output format validation tests.

    Verifies:
    - Required keys exist in output dictionary
    - Processing method is correctly identified
    """

    def test_output_has_required_keys_small_image(self):
        """Small images should have all required output keys."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image((448, 448))

        for key in REQUIRED_KEYS_FIXED:
            assert key in result, f"Missing required key: {key}"

    def test_output_has_required_keys_large_image(self):
        """Large images should have all required output keys including tile info."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image((1024, 1024))

        for key in REQUIRED_KEYS_TILED:
            assert key in result, f"Missing required key: {key}"

    def test_processing_method_value(self):
        """Processing method should be a valid string."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)
        result = analyst.calculate_image((800, 800))

        assert "processing_method" in result
        assert isinstance(result["processing_method"], str)
        assert result["processing_method"] in ["fixed_resolution", "tile_based", "native_resolution", "dynamic"]


class TestXxxModelTokenFormula:
    """Token formula verification tests.

    Verifies:
    - Token calculation matches documented formula
    - Use parametrize for multiple test cases
    """

    @pytest.mark.parametrize("input_params,expected_tokens", TOKEN_FORMULA_CASES)
    def test_token_formula(self, input_params, expected_tokens):
        """Verify token formula matches expected values.

        Update this test to match your model's formula.
        """
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)

        # Example: If your model has _calculate_tokens method
        # calculated = analyst._calculate_tokens(*input_params)
        # assert calculated == expected_tokens

        # Or test via calculate_image with specific sizes that produce known tokens
        pass  # TODO: Implement model-specific formula test


class TestXxxModelVideo:
    """Video support tests.

    If video is not supported, verify NotImplementedError is raised.
    If video is supported, add appropriate tests.
    """

    def test_video_support(self):
        """Test video support or verify NotImplementedError."""
        from vt_calculator.analysts import load_analyst

        analyst = load_analyst(MODEL_NAME)

        # If video is NOT supported:
        with pytest.raises(NotImplementedError):
            analyst.calculate_video({"width": 1920, "height": 1080, "duration": 10.0})

        # If video IS supported, replace above with actual tests
