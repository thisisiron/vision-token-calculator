# Phi-4-Multimodal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Phi-4-Multimodal vision token calculation support

**Architecture:** Implement `Phi4MultimodalAnalyst` class that calculates tokens based on dynamic HD cropping. Images are processed as global (448x448) + HD crops. Token count depends on crop grid dimensions.

**Tech Stack:** Python, existing `get_optimal_tiled_canvas` for aspect ratio calculation

---

## Token Calculation Formula

```
num_tokens = 256 + 1 + (h_crops * w_crops * 256) + (h_crops * 16) + 16
           = 273 + 256 * h_crops * w_crops + 16 * h_crops
```

Where:
- **256**: Global image tokens (16x16 grid from 448/14/2)
- **1**: Separator token
- **h_crops * w_crops * 256**: HD patch tokens (16x16 per crop)
- **h_crops * 16**: Row-level tokens
- **16**: Fixed overhead

**Constants:**
- `image_size = 448`
- `patch_size = 14`
- `downsample_ratio = 2`
- `min_crops = 1`
- `max_crops = 36`

---

### Task 1: Add Phi4MultimodalAnalyst Class

**Files:**
- Modify: `src/vt_calculator/analysts/analyst.py`
- Test: `tests/test_phi4_multimodal.py`

**Step 1: Write the failing test**

```python
# tests/test_phi4_multimodal.py
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

    def test_2x2_crop_image(self):
        """Image ~896x896: 2x2 crops.

        tokens = 273 + 256*2*2 + 16*2 = 273 + 1024 + 32 = 1329
        """
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((896, 896))

        assert result["image_token"][1] == 1329
        assert result["crop_grid"] == (2, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_phi4_multimodal.py -v`
Expected: FAIL with "cannot import name 'Phi4MultimodalAnalyst'"

**Step 3: Write minimal implementation**

```python
# Add to src/vt_calculator/analysts/analyst.py

class Phi4MultimodalAnalyst(VLMAnalyst):
    """Phi-4-Multimodal vision token calculator.

    Token formula:
        num_tokens = 273 + 256 * h_crops * w_crops + 16 * h_crops

    Where crops are determined by image dimensions / 448.
    """

    IMAGE_SIZE = 448
    PATCH_SIZE = 14
    DOWNSAMPLE_RATIO = 2
    MIN_CROPS = 1
    MAX_CROPS = 36

    def __init__(self):
        super().__init__(processor=None)
        self.image_token = "<|image|>"
        self.image_size = self.IMAGE_SIZE
        self.patch_size = self.PATCH_SIZE
        # After patch + downsample: 448/14/2 = 16
        self.grid_size = self.IMAGE_SIZE // self.PATCH_SIZE // self.DOWNSAMPLE_RATIO

    def _calculate_crop_grid(self, height: int, width: int) -> Tuple[int, int]:
        """Calculate optimal crop grid for given image dimensions."""
        if height <= self.image_size and width <= self.image_size:
            return (1, 1)

        return get_optimal_tiled_canvas(
            original_image_size=(height, width),
            target_tile_size=(self.image_size, self.image_size),
            min_image_tiles=self.MIN_CROPS,
            max_image_tiles=self.MAX_CROPS,
        )

    def _calculate_tokens(self, h_crops: int, w_crops: int) -> int:
        """Calculate total tokens for given crop grid.

        Formula: 273 + 256 * h * w + 16 * h
        - 256: global image (16x16)
        - 1: separator
        - 256 * h * w: HD patches
        - 16 * h: row tokens
        - 16: overhead
        """
        global_tokens = self.grid_size * self.grid_size  # 256
        separator = 1
        hd_tokens = h_crops * w_crops * global_tokens
        row_tokens = h_crops * self.grid_size
        overhead = 16

        return global_tokens + separator + hd_tokens + row_tokens + overhead

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        height, width = image_size
        w_crops, h_crops = self._calculate_crop_grid(height, width)
        total_tokens = self._calculate_tokens(h_crops, w_crops)

        return {
            "image_token": (self.image_token, total_tokens),
            "image_token_format": f"{self.image_token}*{total_tokens}",
            "image_size": (height, width),
            "crop_grid": (w_crops, h_crops),
            "patch_size": self.patch_size,
            "grid_size": self.grid_size,
        }

    def calculate_video(
        self,
        video_metadata: dict,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> dict:
        raise NotImplementedError("Phi-4-Multimodal video not yet supported")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_phi4_multimodal.py::TestPhi4MultimodalBasic -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vt_calculator/analysts/analyst.py tests/test_phi4_multimodal.py
git commit -m "feat: add Phi4MultimodalAnalyst class with token calculation"
```

---

### Task 2: Register Model in Factory

**Files:**
- Modify: `src/vt_calculator/analysts/__init__.py`
- Test: `tests/test_phi4_multimodal.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_phi4_multimodal.py

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_phi4_multimodal.py::TestPhi4MultimodalRegistration -v`
Expected: FAIL with "phi4-multimodal not in SUPPORTED_MODELS"

**Step 3: Update __init__.py**

```python
# Modify src/vt_calculator/analysts/__init__.py

# Add import
from .analyst import (
    # ... existing imports ...
    Phi4MultimodalAnalyst,
)

# Add to MODEL_TO_HF_ID
MODEL_TO_HF_ID: dict[str, Optional[str]] = {
    # ... existing models ...
    "phi4-multimodal": None,  # No processor needed
}

# Add to ANALYST_REGISTRY in load_analyst()
ANALYST_REGISTRY: Dict[str, Tuple[Callable, bool]] = {
    # ... existing entries ...
    "phi4-multimodal": (lambda proc, cfg: Phi4MultimodalAnalyst(), False),
}

# Add to __all__
__all__ = [
    # ... existing exports ...
    "Phi4MultimodalAnalyst",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_phi4_multimodal.py::TestPhi4MultimodalRegistration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vt_calculator/analysts/__init__.py tests/test_phi4_multimodal.py
git commit -m "feat: register phi4-multimodal in model factory"
```

---

### Task 3: Add Comprehensive Tests

**Files:**
- Modify: `tests/test_phi4_multimodal.py`

**Step 1: Write additional tests**

```python
# Add to tests/test_phi4_multimodal.py

class TestPhi4MultimodalCropCalculation:
    """Crop grid calculation tests."""

    def test_wide_image_crops(self):
        """Wide image (1920x1080) should get appropriate crops."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((1080, 1920))

        w_crops, h_crops = result["crop_grid"]
        assert w_crops > h_crops  # wider than tall

    def test_tall_image_crops(self):
        """Tall image (1080x1920) should get appropriate crops."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((1920, 1080))

        w_crops, h_crops = result["crop_grid"]
        assert h_crops > w_crops  # taller than wide

    def test_max_crops_respected(self):
        """Very large images should not exceed max_crops."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((5000, 5000))

        w_crops, h_crops = result["crop_grid"]
        assert w_crops * h_crops <= 36


class TestPhi4MultimodalTokenFormula:
    """Token formula verification tests."""

    import pytest

    @pytest.mark.parametrize(
        "h_crops,w_crops,expected",
        [
            (1, 1, 273 + 256 * 1 + 16 * 1),      # 545
            (2, 2, 273 + 256 * 4 + 16 * 2),      # 1329
            (3, 2, 273 + 256 * 6 + 16 * 3),      # 1857
            (2, 3, 273 + 256 * 6 + 16 * 2),      # 1841
            (3, 3, 273 + 256 * 9 + 16 * 3),      # 2625
        ],
    )
    def test_token_formula(self, h_crops, w_crops, expected):
        """Verify: tokens = 273 + 256*h*w + 16*h."""
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        calculated = analyst._calculate_tokens(h_crops, w_crops)
        assert calculated == expected


class TestPhi4MultimodalOutputFormat:
    """Output format tests."""

    def test_output_has_required_keys(self):
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        result = analyst.calculate_image((1024, 1024))

        required_keys = ["image_token", "image_token_format", "image_size",
                         "crop_grid", "patch_size", "grid_size"]
        for key in required_keys:
            assert key in result

    def test_video_raises_not_implemented(self):
        from vt_calculator.analysts.analyst import Phi4MultimodalAnalyst

        analyst = Phi4MultimodalAnalyst()
        with pytest.raises(NotImplementedError):
            analyst.calculate_video({"width": 1920, "height": 1080})
```

**Step 2: Run all tests**

Run: `pytest tests/test_phi4_multimodal.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_phi4_multimodal.py
git commit -m "test: add comprehensive Phi4Multimodal tests"
```

---

### Task 4: Verify Full Integration

**Files:**
- None (verification only)

**Step 1: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_analysts.py
```

**Step 2: Test CLI integration**

```bash
vt-calc --size 1920 1080 -m phi4-multimodal
vt-calc --size 448 448 -m phi4-multimodal
vt-calc --compare phi4-multimodal,qwen2.5-vl --size 1920 1080
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Phi-4-Multimodal integration"
```

---

## Notes

1. **Token formula may need verification**: The formula is derived from source code analysis. If actual token counts differ, adjust `_calculate_tokens()`.

2. **No HF Processor dependency**: This implementation calculates tokens offline without network access.

3. **Video support**: Currently raises `NotImplementedError`. Can be added later if Phi-4 supports video.
