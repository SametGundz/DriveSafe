#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="driver-distraction-and-drowsiness-detection-system",
    version="0.1.0",
    description="Sürücü uykululuk tespiti için gerçek zamanlı bir sistem",
    author="Samet",
    author_email="samet-gunduz@hotmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.5.64",
        "PyYAML>=6.0",
        "matplotlib>=3.5.1",
        "mediapipe>=0.8.9",
        "scipy>=1.7.3",
        "torch>=1.10.0",
        "torchvision>=0.11.1",
        "tqdm>=4.62.3",
        "argparse>=1.4.0",
        "pandas>=1.3.4",
        "seaborn>=0.11.2",
    ],
    entry_points={
        'console_scripts': [
            'drowsiness-detection=src.main:main',
        ],
    },
    python_requires='>=3.7',
) 