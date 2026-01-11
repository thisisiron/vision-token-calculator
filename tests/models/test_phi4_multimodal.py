"""Phi-4-Multimodal Vision Token Calculator Tests."""

import pytest


class TestPhi4MultimodalBasic:
    """Basic token calculation tests."""

    def test_small_image_no_hd_crops(self):
        """Image <= 448x448: only global view, no HD crops.

        tokens = 256 + 1 + 0 + 0 + 16 = 273
        """
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((448, 448))

        assert result["image_token"][1] == 273
        assert result["processing_method"] == "fixed_resolution"

    def test_2x2_crop_image(self):
        """Image ~896x896: 2x2 crops.

        tokens = 273 + 256*2*2 + 16*2 = 273 + 1024 + 32 = 1329
        """
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((896, 896))

        assert result["image_token"][1] == 1329
        assert result["tile_grid"] == (2, 2)
        assert result["processing_method"] == "tile_based"


class TestPhi4MultimodalRegistration:
    """Model registration tests."""

    def test_model_in_supported_models(self):
        from vt_calculator.analysts import SUPPORTED_MODELS

        assert "phi4-multimodal" in SUPPORTED_MODELS

    def test_load_analyst_returns_correct_class(self):
        from vt_calculator.analysts import load_analyst
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = load_analyst("phi4-multimodal")
        assert isinstance(analyst, Phi4MultimodalAnalyst)


class TestPhi4MultimodalCropCalculation:
    """Crop grid calculation tests."""

    def test_wide_image_crops(self):
        """Wide image (1920x1080) should get appropriate crops."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((1080, 1920))

        h_crops, w_crops = result["tile_grid"]
        assert w_crops > h_crops  # wider than tall

    def test_tall_image_crops(self):
        """Tall image (1080x1920) should get appropriate crops."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((1920, 1080))

        h_crops, w_crops = result["tile_grid"]
        assert h_crops > w_crops  # taller than wide

    def test_max_crops_respected(self):
        """Very large images should not exceed max_crops."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((5000, 5000))

        h_crops, w_crops = result["tile_grid"]
        assert h_crops * w_crops <= 36


class TestPhi4MultimodalTokenFormula:
    """Token formula verification tests."""

    @pytest.mark.parametrize(
        "h_crops,w_crops,expected",
        [
            (1, 1, 273),  # No HD processing for 1x1
            (2, 2, 273 + 256 * 4 + 16 * 2),  # 1329
            (3, 2, 273 + 256 * 6 + 16 * 3),  # 1857
            (2, 3, 273 + 256 * 6 + 16 * 2),  # 1841
            (3, 3, 273 + 256 * 9 + 16 * 3),  # 2625
        ],
    )
    def test_token_formula(self, h_crops, w_crops, expected):
        """Verify token formula. 1x1 gets no HD, others get 273 + 256*h*w + 16*h."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        calculated = analyst._calculate_tokens(h_crops, w_crops)
        assert calculated == expected


class TestPhi4MultimodalOutputFormat:
    """Output format tests."""

    def test_output_has_required_keys_small_image(self):
        """Small images use fixed_resolution method."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((448, 448))

        required_keys = [
            "processing_method",
            "image_token",
            "image_token_format",
            "image_size",
            "resized_size",
            "patch_size",
            "patch_grid",
            "total_patches",
            "has_global_patch",
        ]
        for key in required_keys:
            assert key in result
        assert result["processing_method"] == "fixed_resolution"

    def test_output_has_required_keys_large_image(self):
        """Large images use tile_based method."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((1024, 1024))

        required_keys = [
            "processing_method",
            "image_token",
            "image_token_format",
            "image_size",
            "resized_size",
            "tile_size",
            "tile_grid",
            "number_of_tiles",
            "has_global_patch",
            "patch_size",
            "patches_per_tile",
            "total_patches",
        ]
        for key in required_keys:
            assert key in result
        assert result["processing_method"] == "tile_based"

    def test_video_raises_not_implemented(self):
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        with pytest.raises(NotImplementedError):
            analyst.calculate_video({"width": 1920, "height": 1080})
