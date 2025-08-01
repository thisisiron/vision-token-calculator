from ..utils import smart_resize


class VLMAnalyst:
    def __init__(self, processor):
        self.processor = processor

    def get_num_image_patches(self):
        pass

    def get_num_image_tokens(self):
        pass


class Qwen2VLAnalyst(VLMAnalyst):
    def __init__(self, processor):
        super().__init__(processor)

        self.image_token: str = "<|image_pad|>"
        self.image_start_token: str = "<|im_start|>"
        self.image_end_token: str = "<|im_end|>"

        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels

    def get_num_patches(self, image_size):
        height, width = image_size
        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, self.min_pixels, self.max_pixels
        )
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
        return grid_h * grid_w

    def get_num_image_tokens(self, image_size):
        num_patches = self.get_num_patches(image_size)
        return num_patches // self.merge_size**2


class Qwen2_5_VLAnalyst(Qwen2VLAnalyst):
    pass


__all__ = [
    "Qwen2VLAnalyst",
    "Qwen2_5_VLAnalyst",
]
