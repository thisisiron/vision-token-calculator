from typing import Tuple

from .tools import (
    resize_and_grid,
    get_optimal_tiled_canvas,
    select_best_resolution,
    get_patch_output_size,
    get_padding_size,
    get_unpadded_features,
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
            "number_of_image_patches": num_tokens,
            "patch_size": self.patch_size,
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

        return {
            "number_of_image_patches": num_patches,
            "patch_size": self.patch_size,
            "has_global_patch": False,
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

        return {
            "number_of_image_patches": num_patches,
            "patch_size": self.patch_size,
            "has_global_patch": False,
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
            "number_of_image_patches": num_patches,
            "grid_size": (grid_h, grid_w),
            "patch_size": self.patch_size,
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
        
        num_video_tokens = (
            tokens_per_frame * num_frames
        ) // self.temporal_patch_size

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
    pass


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

        return {
            "number_of_image_patches": num_patches,
            "grid_size": (grid_h, grid_w),
            "tile_size": self.tile_size,
            "patch_size": self.patch_size,
            "has_global_patch": num_patches > 1,
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
