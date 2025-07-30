from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vt-calc",
    version="0.0.2",
    author="Vision Token Calculator",
    description="Calculate the number of tokens used for images in vision language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thisisiron/vision-token-calculator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "qwen-vl-utils>=0.0.8",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "vt-calc=vt_calculator.core.calculator:main",
        ],
    },
    keywords=[
        "vision",
        "tokens",
        "language model",
        "multimodal",
        "ai",
        "vlm",
        "vision language model",
        "vision language model token calculator",
    ],
)
