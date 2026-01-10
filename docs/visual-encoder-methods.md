# Visual Encoder Processing Methods in Vision Language Models

## Overview

Vision Language Models (VLMs) process images through a visual encoder before feeding them to the language model. Traditional Vision Transformers (ViT) require fixed-resolution inputs (e.g., 224×224 or 336×336), which forces images to be resized and potentially distorts their aspect ratio or loses fine details.

Two main paradigms have emerged to handle arbitrary image resolutions:

1. **Native Resolution (NaViT-style)**: Process images at their original resolution by directly splitting into patches
2. **Tile-based (AnyRes-style)**: Divide images into fixed-size tiles that match the ViT's native input size

This document explains both approaches, their algorithms, token calculation formulas, and trade-offs.

---

## Native Resolution (NaViT-style)

### Concept

The Native Resolution approach processes images at their original resolution without forcing them into a fixed size. Instead of resizing, the image is directly split into patches matching the ViT's patch size (e.g., 14×14 pixels).

Key innovations from NaViT (Patch n' Pack):
- **Sequence packing**: Multiple images can be packed into a single sequence during training
- **Factorized positional embeddings**: Enable variable aspect ratios and extrapolation to unseen resolutions
- **Continuous token dropping**: Vary the token dropping rate per-image for training efficiency

### Models Using This Approach

- **Qwen2-VL / Qwen2.5-VL**: Implements "Naive Dynamic Resolution" with 2D-RoPE (Rotary Position Embedding)
- **Qwen3-VL**: Extends the approach with M-RoPE (Multimodal RoPE) for text, image, and video

### Algorithm

1. Receive input image with dimensions (H, W)
2. Apply resolution constraints (min_pixels, max_pixels)
3. Resize to nearest valid dimensions (multiples of patch_size × merge_factor)
4. Split into non-overlapping patches
5. Apply 2D positional embeddings
6. Merge adjacent patches to reduce sequence length

### Token Calculation

Given:
- Image size: (H, W)
- Patch size: P (typically 14)
- Merge factor: M (typically 2, merging 2×2 patches into 1 token)

The number of visual tokens is:

$$
N_{tokens} = \left\lceil \frac{H}{P \times M} \right\rceil \times \left\lceil \frac{W}{P \times M} \right\rceil
$$

For Qwen2-VL with P=14 and M=2, the effective factor is 28:

$$
N_{tokens} = \left\lceil \frac{H}{28} \right\rceil \times \left\lceil \frac{W}{28} \right\rceil
$$

---

## Tile-based (AnyRes / Dynamic High-Resolution)

### Concept

The Tile-based approach divides high-resolution images into multiple fixed-size tiles, where each tile matches the ViT's native input resolution (e.g., 336×336 or 448×448). This allows reusing pretrained ViT models without architectural modifications.

Most implementations include a **global thumbnail** (the full image resized to tile size) to provide overall context alongside the detailed tiles.

### Models Using This Approach

- **LLaVA-NeXT**: AnyRes with predefined grid configurations {2×2, 1×{2,3,4}, {2,3,4}×1}
- **LLaVA-OneVision**: Extended AnyRes supporting up to 6×6 grids
- **InternVL2 / InternVL2.5**: Dynamic tile allocation with configurable n_max parameter

### Algorithm

1. Define allowed grid configurations (e.g., 1×1, 1×2, 2×1, 2×2, 2×3, ...)
2. For each configuration, calculate the resulting canvas size
3. Select the configuration that:
   - Best matches the original aspect ratio
   - Minimizes wasted pixels from padding
   - Stays within the maximum tile budget
4. Resize image to fit the selected grid
5. Split into tiles and add global thumbnail
6. Process each tile through ViT independently

### Grid Selection Algorithm (LLaVA-NeXT)

```python
def select_best_resolution(image_size, possible_resolutions):
    original_h, original_w = image_size
    original_aspect = original_w / original_h

    best_resolution = None
    best_fit = infinity

    for (canvas_h, canvas_w) in possible_resolutions:
        canvas_aspect = canvas_w / canvas_h

        # Calculate scale to fit image in canvas
        if original_aspect > canvas_aspect:
            scale = canvas_w / original_w
        else:
            scale = canvas_h / original_h

        # Calculate effective resolution and wasted pixels
        effective_h = min(floor(original_h * scale), canvas_h)
        effective_w = min(floor(original_w * scale), canvas_w)
        wasted = (canvas_h * canvas_w) - (effective_h * effective_w)

        if wasted < best_fit:
            best_fit = wasted
            best_resolution = (canvas_h, canvas_w)

    return best_resolution
```

### Token Calculation

**Basic formula:**

Given:
- Grid dimensions: (G_h, G_w)
- Tokens per tile: T
- Global patch: included (+1 tile)

$$
N_{tokens} = (G_h \times G_w + 1) \times T
$$

**LLaVA-NeXT with unpadding:**

LLaVA-NeXT applies "unpadding" to remove tokens corresponding to padded regions:

$$
N_{tokens} = \sum_{i=1}^{G_h \times G_w} T_{unpadded,i} + T_{global} + N_{newline}
$$

Where:
- $T_{unpadded,i}$: Actual tokens for tile i after removing padding
- $T_{global}$: Tokens for global thumbnail
- $N_{newline}$: Newline tokens between tile rows (for spatial structure)

**InternVL2.5:**

$$
N_{tokens} = N_{patches} \times \left( \frac{tile\_size}{patch\_size \times pixel\_unshuffle} \right)^2
$$

Where:
- $N_{patches}$: Number of tiles (1 for small images, grid_h × grid_w + 1 for larger)
- Typical values: tile_size=448, patch_size=14, pixel_unshuffle=2

---

## References

1. Dehghani, M., et al. "Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution." arXiv:2307.06304 (2023). https://arxiv.org/abs/2307.06304

2. Wang, P., et al. "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution." arXiv:2409.12191 (2024). https://arxiv.org/abs/2409.12191

3. Liu, H., et al. "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge." LLaVA Blog (2024). https://llava-vl.github.io/blog/2024-01-30-llava-next/

4. Chen, Z., et al. "InternVL2.5: A Multimodal Large Language Model for General Visual Understanding." (2024). https://internvl.github.io/blog/2024-12-05-InternVL-2.5/
