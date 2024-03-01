#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="uqocp",
    version="0.1",
    description="Uncertainty quantification methods and tools for OCP models",
    url="https://github.com/jmusiel/uqocp",
    author="Joseph Musielewicz",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    long_description="""Uncertainty quantification methods and tools for OCP models""",
)