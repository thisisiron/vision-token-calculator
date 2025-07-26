import os
import warnings


def setup_quiet_environment():
    warnings.filterwarnings("ignore")

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:
        pass

    try:
        import transformers

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_default_handler()
        transformers.logging.disable_propagation()
    except ImportError:
        pass


if __name__ == "__main__":
    setup_quiet_environment()
    print("Environment configured to minimize Hugging Face warnings.")
