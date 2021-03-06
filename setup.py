#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yolov3",
    version="1.0.0",
    author="Timo Schwarzer",
    author_email="timo.schwarzer@festo.com",
    description="Yolo v3",    
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.festo.com/group/de/cms/10156.htm",
    packages=setuptools.find_packages('include'),    
    package_dir={'':'include'},    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[        
    ],
)