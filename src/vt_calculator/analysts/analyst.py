import math
from typing import Tuple

from .tools import (
    resize_and_grid,
    get_optimal_tiled_canvas,
    select_best_resolution,
    get_patch_output_size,
    get_padding_size,
    get_unpadded_features,
    smart_resize_video,
)


class VLMAnalyst:
    def __init__(self, processor):
        self.processor = processor

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        """
        Calculate the number of image tokens for a given image size.

        Args:
            image_size (Tuple[int, int]): The size of the image in (height, width) format.

        Returns:
            dict: A dictionary containing the number of image tokens and other relevant information.
        """
        raise NotImplementedError

    def calculate_video(
        self,
        video_metadata: dict,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> dict:
        """
        Calculate the number of video tokens.

        Args:
            video_metadata: dict with keys 'width', 'height', 'duration', 'total_frames'
            fps: Target FPS for sampling
            max_frames: Maximum number of frames to use

        Returns:
            dict: Token analysis results
        """
        raise NotImplementedError


class LLaVAAnalyst(VLMAnalyst):
    def __init__(self, processor):
        super().__init__(processor)

        self.image_token: str = "<image>"

        self.resized_height, self.resized_width = (
            processor.image_processor.crop_size["height"],
            processor.image_processor.crop_size["width"],
        )  # (336, 336)

        self.patch_size = processor.patch_size
        self.num_additional_image_tokens = (
            processor.num_additional_image_tokens
        )  # such as CLS (+1)
        self.vision_feature_select_strategy = processor.vision_feature_select_strategy

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        num_tokens = (self.resized_height // self.patch_size) * (
            self.resized_width // self.patch_size
        ) + self.num_additional_image_tokens

        if self.vision_feature_select_strategy == "default":
            num_tokens -= 1  # CLS token is excluded in the default strategy

        return {
            "processing_method": "fixed_resolution",
            "number_of_image_patches": num_tokens,
            "patch_size": self.patch_size,
            "patch_grid": (
                self.resized_height // self.patch_size,
                self.resized_width // self.patch_size,
            ),
            "has_global_patch": False,
            "image_size": image_size,
            "resized_size": (self.resized_height, self.resized_width),
            "image_token": (self.image_token, num_tokens),
            "image_token_format": f"{self.image_token}*{num_tokens}",
        }


class LLaVANextAnalyst(VLMAnalyst):
    def __init__(self, processor):
        super().__init__(processor)

        self.image_token: str = "<image>"

        size = processor.image_processor.size
        self.tile_size = (
            (size["shortest_edge"], size["shortest_edge"])
            if "shortest_edge" in size
            else (
                min(size["height"], size["width"]),
                min(size["height"], size["width"]),
            )
        )  # (336, 336)

        self.patch_size = processor.patch_size
        self.grid_pinpoints = processor.image_processor.image_grid_pinpoints
        self.num_additional_image_tokens = (
            processor.num_additional_image_tokens
        )  # such as CLS (+1)
        self.vision_feature_select_strategy = processor.vision_feature_select_strategy

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        best_resolution = select_best_resolution(image_size, self.grid_pinpoints)
        resized_height, resized_width = get_patch_output_size(
            image_size, best_resolution
        )
        padding_y, padding_x = get_padding_size(
            (resized_height, resized_width), best_resolution
        )

        num_patches = (
            best_resolution[0]
            // self.tile_size[0]
            * best_resolution[1]
            // self.tile_size[1]
            + 1  # global patch
        )

        scale_height, scale_width = (
            best_resolution[0] // self.tile_size[0],
            best_resolution[1] // self.tile_size[1],
        )

        patches_height = self.tile_size[0] // self.patch_size
        patches_width = self.tile_size[1] // self.patch_size

        unpadded_features, newline_features = get_unpadded_features(
            image_size[0],
            image_size[1],
            patches_height,
            patches_width,
            scale_height,
            scale_width,
        )

        base_features = (
            patches_height * patches_width + self.num_additional_image_tokens
        )
        num_image_tokens = unpadded_features + newline_features + base_features

        if self.vision_feature_select_strategy == "default":
            num_image_tokens -= 1  # CLS token is excluded in the default strategy

        patches_per_tile = patches_height * patches_width
        total_patches = num_patches * patches_per_tile

        return {
            "processing_method": "tile_based",
            "tile_size": self.tile_size[0],
            "tile_grid": (scale_height, scale_width),
            "number_of_tiles": num_patches,
            "has_global_patch": True,
            "patch_size": self.patch_size,
            "patches_per_tile": patches_per_tile,
            "total_patches": total_patches,
            "image_size": image_size,
            "resized_size": (resized_height, resized_width),
            "image_token": (self.image_token, num_image_tokens),
            "image_token_format": f"{self.image_token}*{num_image_tokens}",
        }

    def calculate_video(
        self,
        video_metadata: dict,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> dict:
        width = video_metadata["width"]
        height = video_metadata["height"]
        duration = video_metadata["duration"]

        target_fps = fps if fps else 1.0
        num_frames = int(duration * target_fps)
        if num_frames == 0:
            num_frames = 1

        if max_frames and num_frames > max_frames:
            num_frames = max_frames
            target_fps = num_frames / duration if duration > 0 else target_fps

        frame_result = self.calculate_image((height, width))
        tokens_per_frame = frame_result["image_token"][1]

        total_tokens = tokens_per_frame * num_frames

        return {
            "type": "video",
            "number_of_video_tokens": total_tokens,
            "sampled_frames": num_frames,
            "fps": target_fps,
            "duration": duration,
            "grid_size": (0, 0),
            "resized_size": frame_result["resized_size"],
            "image_token": (self.image_token, total_tokens),
            "token_format": f"({self.image_token}...) * {num_frames}",
        }


class LlavaOnevisionAnalyst(VLMAnalyst):
    def __init__(self, processor, config):
        super().__init__(processor)

        self.image_token: str = "<image>"

        size = processor.image_processor.size
        self.tile_size = (
            (size["shortest_edge"], size["shortest_edge"])
            if "shortest_edge" in size
            else (
                min(size["height"], size["width"]),
                min(size["height"], size["width"]),
            )
        )  # (384, 384)

        self.patch_size = config.vision_config.patch_size
        self.grid_pinpoints = processor.image_processor.image_grid_pinpoints
        self.vision_feature_select_strategy = processor.vision_feature_select_strategy
        self.max_num_patches = int(processor.vision_aspect_ratio.strip("anyres_max_"))

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        best_resolution = select_best_resolution(image_size, self.grid_pinpoints)
        resized_height, resized_width = get_patch_output_size(
            image_size, best_resolution
        )
        padding_y, padding_x = get_padding_size(
            (resized_height, resized_width), best_resolution
        )

        num_patches = (
            best_resolution[0]
            // self.tile_size[0]
            * best_resolution[1]
            // self.tile_size[1]
            + 1  # global patch
        )

        scale_height, scale_width = (
            best_resolution[0] // self.tile_size[0],
            best_resolution[1] // self.tile_size[1],
        )

        patches_height = self.tile_size[0] // self.patch_size
        patches_width = self.tile_size[1] // self.patch_size

        unpadded_features, newline_features = get_unpadded_features(
            image_size[0],
            image_size[1],
            patches_height,
            patches_width,
            scale_height,
            scale_width,
            max_num_patches=self.max_num_patches,
        )

        # The base patch covers the entire image (no CLS for SigLIP)
        base_features = patches_height * patches_width
        num_image_tokens = unpadded_features + newline_features + base_features

        if self.vision_feature_select_strategy == "default":
            num_image_tokens -= 1

        patches_per_tile = patches_height * patches_width
        total_patches = num_patches * patches_per_tile

        return {
            "processing_method": "tile_based",
            "tile_size": self.tile_size[0],
            "tile_grid": (scale_height, scale_width),
            "number_of_tiles": num_patches,
            "has_global_patch": True,
            "patch_size": self.patch_size,
            "patches_per_tile": patches_per_tile,
            "total_patches": total_patches,
            "image_size": image_size,
            "resized_size": (resized_height, resized_width),
            "image_token": (self.image_token, num_image_tokens),
            "image_token_format": f"{self.image_token}*{num_image_tokens}",
        }


class Qwen2VLAnalyst(VLMAnalyst):
    def __init__(self, processor):
        super().__init__(processor)

        self.image_token: str = "<|image_pad|>"
        self.image_start_token: str = "<|vision_start|>"
        self.image_end_token: str = "<|vision_end|>"
        self.video_token: str = "<|video_pad|>"

        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        self.min_pixels = (
            processor.image_processor.min_pixels
            or processor.image_processor.size["shortest_edge"]
        )
        self.max_pixels = (
            processor.image_processor.max_pixels
            or processor.image_processor.size["longest_edge"]
        )
        self.temporal_patch_size = 2

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        resized_h, resized_w, grid_h, grid_w = resize_and_grid(
            image_size,
            self.patch_size,
            self.merge_size,
            self.min_pixels,
            self.max_pixels,
        )
        num_patches = grid_h * grid_w

        # Qwen2-VL: merged tokens = patches / (merge_size^2)
        num_tokens = num_patches // (self.merge_size**2)

        return {
            "processing_method": "native_resolution",
            "patch_size": self.patch_size,
            "patch_grid": (grid_h, grid_w),
            "total_patches": num_patches,
            "has_global_patch": False,
            "image_size": image_size,
            "resized_size": (resized_h, resized_w),
            "image_token": (self.image_token, num_tokens),
            "image_start_token": (self.image_start_token, 1),
            "image_end_token": (self.image_end_token, 1),
            "image_token_format": f"{self.image_start_token}{self.image_token}*{num_tokens}{self.image_end_token}",
        }

    def calculate_video(
        self,
        video_metadata: dict,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> dict:
        width = video_metadata["width"]
        height = video_metadata["height"]
        duration = video_metadata["duration"]

        target_fps = fps if fps else 2.0

        num_frames = int(duration * target_fps)
        if num_frames == 0:
            num_frames = 1

        if max_frames and num_frames > max_frames:
            num_frames = max_frames

        resized_h, resized_w, grid_h, grid_w = resize_and_grid(
            (height, width),
            self.patch_size,
            self.merge_size,
            self.min_pixels,
            self.max_pixels,
        )

        tokens_per_frame = (grid_h * grid_w) // (self.merge_size**2)

        num_video_tokens = (tokens_per_frame * num_frames) // self.temporal_patch_size

        return {
            "type": "video",
            "number_of_video_tokens": num_video_tokens,
            "sampled_frames": num_frames,
            "fps": target_fps,
            "duration": duration,
            "grid_size": (grid_h, grid_w),
            "resized_size": (resized_h, resized_w),
            "image_token": (self.video_token, num_video_tokens),
            "image_start_token": (self.image_start_token, 1),
            "image_end_token": (self.image_end_token, 1),
            "token_format": f"{self.image_start_token}{self.video_token}*{num_video_tokens}{self.image_end_token}",
        }


class Qwen2_5_VLAnalyst(Qwen2VLAnalyst):
    pass


class Qwen3VLAnalyst(Qwen2VLAnalyst):
    ASSUMED_SOURCE_FPS = 24.0

    def __init__(self, processor):
        super().__init__(processor)
        self.min_frames = 4
        self.max_frames = 768
        self.video_min_pixels = processor.video_processor.size["shortest_edge"]
        self.video_max_pixels = processor.video_processor.size["longest_edge"]
        self.target_fps = processor.video_processor.fps

    def calculate_video(
        self,
        video_metadata: dict,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> dict:
        width = video_metadata["width"]
        height = video_metadata["height"]
        duration = video_metadata["duration"]
        source_fps = video_metadata.get("fps", self.ASSUMED_SOURCE_FPS)
        total_frames = video_metadata.get("total_frames", int(duration * source_fps))

        sampling_fps = fps if fps else self.target_fps
        extracted_frames = int(total_frames / source_fps * sampling_fps)
        extracted_frames = max(extracted_frames, 1)

        if max_frames and extracted_frames > max_frames:
            extracted_frames = max_frames

        num_frames = int(extracted_frames / self.ASSUMED_SOURCE_FPS * self.target_fps)
        num_frames = min(
            max(num_frames, self.min_frames), self.max_frames, extracted_frames
        )

        factor = self.patch_size * self.merge_size
        resized_h, resized_w = smart_resize_video(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )

        t_bar = (
            math.ceil(num_frames / self.temporal_patch_size) * self.temporal_patch_size
        )
        grid_t = t_bar // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size

        num_video_tokens = grid_t * grid_h * grid_w // (self.merge_size**2)

        return {
            "type": "video",
            "number_of_video_tokens": num_video_tokens,
            "sampled_frames": num_frames,
            "fps": sampling_fps,
            "duration": duration,
            "grid_size": (grid_h, grid_w),
            "grid_t": grid_t,
            "resized_size": (resized_h, resized_w),
            "image_token": (self.video_token, num_video_tokens),
            "image_start_token": (self.image_start_token, 1),
            "image_end_token": (self.image_end_token, 1),
            "token_format": f"{self.image_start_token}{self.video_token}*{num_video_tokens}{self.image_end_token}",
        }


class InternVLAnalyst(VLMAnalyst):
    def __init__(self, processor, config):
        super().__init__(processor)

        self.image_token: str = "<IMG_CONTEXT>"
        self.image_start_token: str = "<img>"
        self.image_end_token: str = "</img>"

        self.min_patches = processor.image_processor.min_patches
        self.max_patches = processor.image_processor.max_patches
        assert (
            processor.image_processor.size["height"]
            == processor.image_processor.size["width"]
        )
        self.tile_size = processor.image_processor.size["height"]

        assert config.vision_config.patch_size[0] == config.vision_config.patch_size[1]
        self.patch_size = config.vision_config.patch_size[0]
        self.pixel_unshuffle_size = 2

        self.image_seq_length = (
            self.tile_size // self.patch_size // self.pixel_unshuffle_size
        ) ** 2

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        num_patches = 1
        grid_w, grid_h = get_optimal_tiled_canvas(
            image_size,
            (self.tile_size, self.tile_size),
            self.min_patches,
            self.max_patches,
        )
        if grid_w * grid_h > 1:
            num_patches += grid_h * grid_w

        num_tokens = num_patches * self.image_seq_length
        patches_per_tile = self.image_seq_length
        total_patches = num_patches * patches_per_tile

        return {
            "processing_method": "tile_based",
            "tile_size": self.tile_size,
            "tile_grid": (grid_h, grid_w),
            "number_of_tiles": num_patches,
            "has_global_patch": num_patches > 1,
            "patch_size": self.patch_size,
            "patches_per_tile": patches_per_tile,
            "total_patches": total_patches,
            "image_size": image_size,
            "resized_size": (self.tile_size * grid_h, self.tile_size * grid_w),
            "image_token": (self.image_token, num_tokens),
            "image_start_token": (self.image_start_token, 1),
            "image_end_token": (self.image_end_token, 1),
            "image_token_format": f"{self.image_start_token}{self.image_token}*{self.image_seq_length}{self.image_token}*{self.image_seq_length}...{self.image_end_token}",
        }

    def calculate_video(
        self,
        video_metadata: dict,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> dict:
        width = video_metadata["width"]
        height = video_metadata["height"]
        duration = video_metadata["duration"]

        target_fps = fps if fps else 1.0
        num_frames = int(duration * target_fps)
        if num_frames == 0:
            num_frames = 1

        if max_frames and num_frames > max_frames:
            num_frames = max_frames
            target_fps = num_frames / duration if duration > 0 else target_fps

        frame_result = self.calculate_image((height, width))
        tokens_per_frame = frame_result["image_token"][1]

        total_tokens = (tokens_per_frame + 2) * num_frames

        return {
            "type": "video",
            "number_of_video_tokens": total_tokens,
            "sampled_frames": num_frames,
            "fps": target_fps,
            "duration": duration,
            "grid_size": frame_result["grid_size"],
            "resized_size": frame_result["resized_size"],
            "image_token": (self.image_token, total_tokens),
            "token_format": f"({self.image_start_token}...{self.image_end_token}) * {num_frames}",
        }


class DeepSeekOCRAnalyst(VLMAnalyst):
    MODES = {
        "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    }

    PATCH_SIZE = 16
    DOWNSAMPLE_RATIO = 4
    MIN_CROPS = 2
    MAX_CROPS = 9

    def __init__(self, mode: str = "base"):
        super().__init__(processor=None)

        if mode not in self.MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from {list(self.MODES.keys())}"
            )

        self.mode = mode
        self.image_token = "<image>"
        self.patch_size = self.PATCH_SIZE
        self.downsample_ratio = self.DOWNSAMPLE_RATIO

        config = self.MODES[mode]
        self.base_size = config["base_size"]
        self.image_size = config["image_size"]
        self.crop_mode = config["crop_mode"]

    def _calculate_num_row(self, size: int) -> int:
        """
        Calculate the grid dimension (visual tokens per row/column) after downsampling.

        For an image of given size:
        - Divide by patch_size to get number of patches per dimension
        - Divide by downsample_ratio to get final grid dimension
        """
        return math.ceil((size // self.patch_size) / self.downsample_ratio)

    def _calculate_native_tokens(self, num_row: int) -> int:
        """
        Calculate tokens for native (non-crop) mode.

        Token layout:
        - Each row: num_row <image> tokens + 1 <image_newline>
        - Total rows: num_row
        - End: 1 <image_seperator>

        Formula: (num_row + 1) * num_row + 1
        Example (tiny, num_row=8): (8 + 1) * 8 + 1 = 73 tokens
        """
        return (num_row + 1) * num_row + 1

    def _calculate_local_tokens(
        self, num_row: int, width_tiles: int, height_tiles: int
    ) -> int:
        """
        Calculate tokens for local crops in gundam mode.

        Token layout:
        - Each row: (num_row * width_tiles) <image> tokens + 1 <image_newline>
        - Total rows: num_row * height_tiles

        Formula: (num_row * width_tiles + 1) * (num_row * height_tiles)
        """
        return (num_row * width_tiles + 1) * (num_row * height_tiles)

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        height, width = image_size

        if self.crop_mode:
            return self._calculate_gundam_mode(height, width)
        else:
            return self._calculate_native_mode(height, width)

    def _calculate_native_mode(self, height: int, width: int) -> dict:
        num_row = self._calculate_num_row(self.image_size)
        num_patches = (self.image_size // self.patch_size) ** 2

        # Calculate individual token counts
        num_image_tokens = num_row * num_row  # Pure <image> tokens
        num_newline_tokens = num_row  # <image_newline> per row
        num_separator_tokens = 1  # <image_separator> at end
        total_tokens = num_image_tokens + num_newline_tokens + num_separator_tokens

        # Format: (<image>*N + <image_newline>) * N + <image_seperator>
        token_format = (
            f"({self.image_token}*{num_row} + <image_newline>) "
            f"* {num_row} + <image_seperator> = {total_tokens}"
        )

        patch_grid = self.image_size // self.patch_size

        return {
            "processing_method": "native_resolution",
            "image_token": (self.image_token, num_image_tokens),
            "image_newline_token": ("<image_newline>", num_newline_tokens),
            "image_separator_token": ("<image_separator>", num_separator_tokens),
            "number_of_image_tokens": total_tokens,
            "image_token_format": token_format,
            "image_size": (height, width),
            "resized_size": (self.image_size, self.image_size),
            "patch_size": self.patch_size,
            "patch_grid": (patch_grid, patch_grid),
            "total_patches": num_patches,
            "has_global_patch": False,
            "mode": self.mode,
            "base_size": self.base_size,
            "num_row": num_row,
        }

    def _calculate_gundam_mode(self, height: int, width: int) -> dict:
        num_row_base = self._calculate_num_row(self.base_size)
        num_row_local = self._calculate_num_row(self.image_size)

        # Global token breakdown
        global_image_tokens = num_row_base * num_row_base
        global_newline_tokens = num_row_base
        global_separator_tokens = 1
        global_tokens = global_image_tokens + global_newline_tokens + global_separator_tokens

        # Determine crop grid: no crops for small images, otherwise use optimal tiling
        if width <= self.image_size and height <= self.image_size:
            crop_grid = (1, 1)
        else:
            crop_grid = get_optimal_tiled_canvas(
                original_image_size=(height, width),
                target_tile_size=(self.image_size, self.image_size),
                min_image_tiles=self.MIN_CROPS,
                max_image_tiles=self.MAX_CROPS,
            )
        width_tiles, height_tiles = crop_grid

        # Local token breakdown
        local_image_tokens = 0
        local_newline_tokens = 0
        local_tokens = 0
        if width_tiles > 1 or height_tiles > 1:
            local_cols = num_row_local * width_tiles
            local_rows = num_row_local * height_tiles
            local_image_tokens = local_cols * local_rows
            local_newline_tokens = local_rows
            local_tokens = self._calculate_local_tokens(
                num_row_local, width_tiles, height_tiles
            )

        # Total breakdown
        total_image_tokens = global_image_tokens + local_image_tokens
        total_newline_tokens = global_newline_tokens + local_newline_tokens
        total_separator_tokens = global_separator_tokens  # Only in global
        total_tokens = global_tokens + local_tokens

        global_patches = (self.base_size // self.patch_size) ** 2
        local_patches = (
            width_tiles * height_tiles * (self.image_size // self.patch_size) ** 2
            if width_tiles > 1 or height_tiles > 1
            else 0
        )
        num_patches = global_patches + local_patches

        # Build token format string
        if local_tokens > 0:
            local_cols = num_row_local * width_tiles
            local_rows = num_row_local * height_tiles
            token_format = (
                f"global({global_tokens}) + "
                f"local(({self.image_token}*{local_cols} + <image_newline>) "
                f"* {local_rows} = {local_tokens}) = {total_tokens}"
            )
        else:
            token_format = (
                f"({self.image_token}*{num_row_base} + <image_newline>) "
                f"* {num_row_base} + <image_seperator> = {total_tokens}"
            )

        has_local_crops = width_tiles > 1 or height_tiles > 1
        global_patches_per_tile = (self.base_size // self.patch_size) ** 2
        local_patches_per_tile = (self.image_size // self.patch_size) ** 2 if has_local_crops else 0
        number_of_tiles = 1 + (width_tiles * height_tiles if has_local_crops else 0)

        return {
            "processing_method": "tile_based",
            "image_token": (self.image_token, total_image_tokens),
            "image_newline_token": ("<image_newline>", total_newline_tokens),
            "image_separator_token": ("<image_separator>", total_separator_tokens),
            "number_of_image_tokens": total_tokens,
            "image_token_format": token_format,
            "image_size": (height, width),
            "resized_size": (self.base_size, self.base_size),
            "tile_size": self.image_size,
            "tile_grid": (height_tiles, width_tiles),
            "number_of_tiles": number_of_tiles,
            "has_global_patch": has_local_crops,
            "patch_size": self.patch_size,
            "patches_per_tile": local_patches_per_tile if has_local_crops else global_patches_per_tile,
            "total_patches": num_patches,
            "mode": self.mode,
            "base_size": self.base_size,
            "num_global_tokens": global_tokens,
            "num_local_tokens": local_tokens,
        }

    def calculate_video(
        self,
        video_metadata: dict,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> dict:
        raise NotImplementedError("DeepSeek-OCR does not support video input")


class Phi4MultimodalAnalyst(VLMAnalyst):
    """Phi-4-Multimodal vision token calculator.

    Token formula:
        num_tokens = 273 + 256 * h_crops * w_crops + 16 * h_crops

    Where:
    - 256: Global image tokens (16x16 grid from 448/14/2)
    - 1: Separator token
    - 256 * h * w: HD patch tokens
    - 16 * h: Row-level tokens
    - 16: Fixed overhead

    Constants:
    - image_size = 448
    - patch_size = 14
    - downsample_ratio = 2
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

        For small images (1x1 crop): 256 + 1 + 16 = 273 (no HD)
        For larger images: 273 + 256 * h * w + 16 * h
        """
        global_tokens = self.grid_size * self.grid_size  # 256
        separator = 1
        overhead = 16

        # No HD processing for 1x1 (small images)
        if h_crops == 1 and w_crops == 1:
            return global_tokens + separator + overhead

        # HD processing for larger images
        hd_tokens = h_crops * w_crops * global_tokens
        row_tokens = h_crops * self.grid_size

        return global_tokens + separator + hd_tokens + row_tokens + overhead

    def calculate_image(self, image_size: Tuple[int, int]) -> dict:
        height, width = image_size
        w_crops, h_crops = self._calculate_crop_grid(height, width)
        total_tokens = self._calculate_tokens(h_crops, w_crops)

        # Determine processing method based on crop grid
        is_tiled = h_crops > 1 or w_crops > 1
        patches_per_tile = self.grid_size * self.grid_size

        if is_tiled:
            return {
                "processing_method": "tile_based",
                "image_token": (self.image_token, total_tokens),
                "image_token_format": f"{self.image_token}*{total_tokens}",
                "image_size": (height, width),
                "resized_size": (self.image_size * h_crops, self.image_size * w_crops),
                "tile_size": self.image_size,
                "tile_grid": (h_crops, w_crops),
                "number_of_tiles": h_crops * w_crops + 1,
                "has_global_patch": True,
                "patch_size": self.patch_size,
                "patches_per_tile": patches_per_tile,
                "total_patches": (h_crops * w_crops + 1) * patches_per_tile,
            }
        else:
            return {
                "processing_method": "fixed_resolution",
                "image_token": (self.image_token, total_tokens),
                "image_token_format": f"{self.image_token}*{total_tokens}",
                "image_size": (height, width),
                "resized_size": (self.image_size, self.image_size),
                "patch_size": self.patch_size,
                "patch_grid": (self.grid_size, self.grid_size),
                "total_patches": patches_per_tile,
                "has_global_patch": False,
            }

    def calculate_video(
        self,
        video_metadata: dict,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> dict:
        raise NotImplementedError("Phi-4-Multimodal video not yet supported")
