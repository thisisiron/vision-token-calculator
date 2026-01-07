"""Video backend selection and availability checking."""

import os
import sys
import logging
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


class VideoBackend(Enum):
    TORCHCODEC = "torchcodec"
    DECORD = "decord"
    TORCHVISION = "torchvision"
    PYAV = "pyav"


def is_torchcodec_available() -> bool:
    try:
        import importlib.util

        if importlib.util.find_spec("torchcodec") is None:
            return False
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return True
    except (ImportError, AttributeError, Exception):
        return False


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def is_torchvision_available() -> bool:
    try:
        import importlib.util

        if importlib.util.find_spec("torchvision") is None:
            return False
        import torchvision.io  # noqa: F401

        return True
    except (ImportError, AttributeError, Exception):
        return False


def is_pyav_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("av") is not None


@lru_cache(maxsize=1)
def get_default_backend() -> VideoBackend:
    forced_backend = os.getenv("FORCE_VIDEO_BACKEND", None)

    if forced_backend is not None:
        try:
            backend = VideoBackend[forced_backend.upper()]
            print(f"Using forced video backend: {backend.value}", file=sys.stderr)
            return backend
        except KeyError:
            logger.warning(
                f"Invalid FORCE_VIDEO_BACKEND value: {forced_backend}. "
                f"Valid options: {[b.name for b in VideoBackend]}"
            )

    if is_torchcodec_available():
        backend = VideoBackend.TORCHCODEC
    elif is_decord_available():
        backend = VideoBackend.DECORD
    elif is_torchvision_available():
        backend = VideoBackend.TORCHVISION
    elif is_pyav_available():
        backend = VideoBackend.PYAV
    else:
        raise RuntimeError(
            "No video backend available. Please install at least one of: "
            "av, torchvision, decord, or torchcodec"
        )

    print(f"Auto-selected video backend: {backend.value}", file=sys.stderr)
    return backend
