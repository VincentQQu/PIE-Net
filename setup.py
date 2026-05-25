#!/usr/bin/env python3
"""Setup script for PIE-Net package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="event-pienet",
    version="1.1.0",
    author="Vincent Qu",
    description="PIE-Net: Probabilistic Intensity-Event Modeling for High Quality Event-Based Video Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VincentQQu/pie-net",
    packages=find_packages(),
    package_data={
        "pie_net": ["pretrained/*.pth"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "realtime": ["dv-processing>=1.7.0"],
        "eval": [
            "scikit-image>=0.19.0",
            "pyiqa>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pie-net-demo=pie_net.demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords="event-camera, video-reconstruction, deep-learning, computer-vision, probabilistic-modeling",
)
