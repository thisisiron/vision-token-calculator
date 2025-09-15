from .analyst import Qwen2VLAnalyst, Qwen2_5_VLAnalyst, InternVLAnalyst

# Mapping from model path (lowercased) to Analyst class
ANALYST_CLASS_BY_MODEL: dict[str, type[Qwen2VLAnalyst]] = {
    # InternVL
    "opengvlab/internvl3-1b-hf": InternVLAnalyst,
    # Qwen2.5-VL family
    "qwen/qwen2.5-vl-3b-instruct": Qwen2_5_VLAnalyst,
    "qwen/qwen2.5-vl-7b-instruct": Qwen2_5_VLAnalyst,
    "qwen/qwen2.5-vl-32b-instruct": Qwen2_5_VLAnalyst,
}


def get_analyst_class_for_model(model_path: str):
    """Return analyst class using dictionary mapping with case-insensitive key.

    Args:
        model_path (str): Hugging Face model path

    Returns:
        type[Qwen2VLAnalyst]: Analyst class to instantiate
    """
    key = model_path.strip().lower()
    return ANALYST_CLASS_BY_MODEL.get(key, Qwen2_5_VLAnalyst)


__all__ = [
    "Qwen2VLAnalyst",
    "Qwen2_5_VLAnalyst",
    "InternVLAnalyst",
    "ANALYST_CLASS_BY_MODEL",
    "get_analyst_class_for_model",
]
