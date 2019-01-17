#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <=9.0.3
    from pip.req import parse_requirements


setup(
    name='mi_models',
    version='0.0.2',
    description='CMS lib for predict market impact',
    author='cms',
    packages=find_packages(exclude=[]),
    zip_safe=False
)
