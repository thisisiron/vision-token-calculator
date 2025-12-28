.PHONY: style

check_dirs := src tests
exclude_folders := __pycache__ .git venv dist

style:
	python -m ruff check $(check_dirs) setup.py --fix --exclude $(exclude_folders)
	python -m ruff format $(check_dirs) setup.py --exclude $(exclude_folders)
