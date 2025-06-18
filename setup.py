from setuptools import setup, find_packages

setup(
    name="vision-token-calculator",
    version="0.0.1",
    author="Vision Token Calculator",
    description="Calculate the number of tokens used for images in vision language models",
    url="https://github.com/thisisiron/vision-token-calculator",
    py_modules=["calculate"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
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
            "vision-token-calculator=calculate:main",
            "vt-calc=calculate:main",
        ],
    },
    keywords=["vision", "tokens", "language model", "multimodal", "ai", "vlm", "vision language model", "vision language model token calculator"],
) 