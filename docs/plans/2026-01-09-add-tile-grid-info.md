# Add Tile Grid Info to Tile-Based Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Display tile grid configuration (e.g., 2×3) for tile-based models like LLaVA-Next, LLaVA-OneVision, and DeepSeek-OCR gundam mode.

**Architecture:** Add `grid_size` and `tile_size` fields to the return dict of tile-based analysts. The reporter already handles these fields when present.

**Tech Stack:** Python, existing analyst/reporter infrastructure

---

## Task 1: Add grid_size and tile_size to LLaVANextAnalyst

**Files:**
- Modify: `src/vt_calculator/analysts/analyst.py:152-160`
- Test: Manual CLI test

**Step 1: Add grid_size and tile_size to return dict**

In `LLaVANextAnalyst.calculate_image()`, add `grid_size` and `tile_size` to the return dict:

```python
return {
    "number_of_image_patches": num_patches,
    "patch_size": self.patch_size,
    "tile_size": self.tile_size[0],
    "grid_size": (scale_height, scale_width),
    "has_global_patch": True,
    "image_size": image_size,
    "resized_size": (resized_height, resized_width),
    "image_token": (self.image_token, num_image_tokens),
    "image_token_format": f"{self.image_token}*{num_image_tokens}",
}
```

Changes:
- Add `"tile_size": self.tile_size[0]` (tile is square, so use height)
- Add `"grid_size": (scale_height, scale_width)`
- Change `"has_global_patch": False` to `True` (LLaVA-Next uses global patch)

**Step 2: Verify with CLI**

Run: `./venv/bin/python -m vt_calculator.cli.main -m llava-next --size 1920 1080 2>&1 | grep -v RuntimeWarning`

Expected: Output should now show "Tile Size" and "Grid Size (H×W)" in PATCH INFO section.

**Step 3: Commit**

```bash
git add src/vt_calculator/analysts/analyst.py
git commit -m "feat(llava-next): add grid_size and tile_size to output"
```

---

## Task 2: Add grid_size and tile_size to LlavaOnevisionAnalyst

**Files:**
- Modify: `src/vt_calculator/analysts/analyst.py:262-270`
- Test: Manual CLI test

**Step 1: Add grid_size and tile_size to return dict**

In `LlavaOnevisionAnalyst.calculate_image()`, add `grid_size` and `tile_size` to the return dict:

```python
return {
    "number_of_image_patches": num_patches,
    "patch_size": self.patch_size,
    "tile_size": self.tile_size[0],
    "grid_size": (scale_height, scale_width),
    "has_global_patch": True,
    "image_size": image_size,
    "resized_size": (resized_height, resized_width),
    "image_token": (self.image_token, num_image_tokens),
    "image_token_format": f"{self.image_token}*{num_image_tokens}",
}
```

Changes:
- Add `"tile_size": self.tile_size[0]`
- Add `"grid_size": (scale_height, scale_width)`
- Change `"has_global_patch": False` to `True`

**Step 2: Verify with CLI**

Run: `./venv/bin/python -m vt_calculator.cli.main -m llava-onevision --size 1920 1080 2>&1 | grep -v RuntimeWarning`

Expected: Output should now show "Tile Size" and "Grid Size (H×W)" in PATCH INFO section.

**Step 3: Commit**

```bash
git add src/vt_calculator/analysts/analyst.py
git commit -m "feat(llava-onevision): add grid_size and tile_size to output"
```

---

## Task 3: Add grid_size to DeepSeek-OCR gundam mode

**Files:**
- Modify: `src/vt_calculator/analysts/analyst.py:631-644`
- Test: Manual CLI test

**Step 1: Add grid_size to gundam mode return dict**

In `DeepSeekOCRAnalyst._calculate_gundam_mode()`, add `grid_size` using the existing `crop_grid`:

```python
return {
    "image_token": (self.image_token, total_tokens),
    "image_token_format": f"{self.image_token}*{total_tokens}",
    "image_size": (height, width),
    "resized_size": (self.base_size, self.base_size),
    "number_of_image_patches": num_patches,
    "has_global_patch": width_tiles > 1 or height_tiles > 1,
    "grid_size": (height_tiles, width_tiles),
    "mode": self.mode,
    "base_size": self.base_size,
    "patch_size": self.patch_size,
    "crop_grid": crop_grid,
    "num_global_tokens": global_tokens,
    "num_local_tokens": local_tokens,
}
```

Change:
- Add `"grid_size": (height_tiles, width_tiles)` (H×W format consistent with other models)

**Step 2: Verify with CLI**

Run: `./venv/bin/python -m vt_calculator.cli.main -m deepseek-ocr-gundam --size 1920 1080 2>&1 | grep -v RuntimeWarning`

Expected: Output should now show "Grid Size (H×W)" in PATCH INFO section.

**Step 3: Commit**

```bash
git add src/vt_calculator/analysts/analyst.py
git commit -m "feat(deepseek-ocr): add grid_size to gundam mode output"
```

---

## Task 4: Run all tests

**Step 1: Run pytest**

Run: `./venv/bin/pytest tests/ -v --tb=short`

Expected: All tests pass.

**Step 2: Final commit (if any fixes needed)**

If tests fail, fix and commit fixes.

---

## Task 5: Push and update PR

**Step 1: Push changes**

```bash
git push origin feature/add-deepseek-ocr
```

**Step 2: Verify PR updated**

Check: https://github.com/thisisiron/vision-token-calculator/pull/17
