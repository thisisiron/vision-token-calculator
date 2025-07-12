#!/usr/bin/env python3
"""
Environment setup script to disable Hugging Face warnings and logging.
Run this before importing any transformers modules.
"""

import os
import warnings
import logging

def setup_quiet_environment():
    """
    Set up environment variables and logging to minimize Hugging Face output.
    """
    # Environment variables to control HF behavior
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    
    # Disable Python warnings
    warnings.filterwarnings('ignore')
    
    # Set logging level to ERROR for common libraries
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    
    # Disable transformers specific logging
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
