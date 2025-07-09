#!/bin/bash

check_dirs="tests"
exclude_folders="__pycache__ .git venv dist"

# ruff check
python -m ruff check $check_dirs setup.py --fix --exclude $exclude_folders

# ruff format  
python -m ruff format $check_dirs setup.py --exclude $exclude_folders