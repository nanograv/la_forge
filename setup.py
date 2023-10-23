#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy>=1.16',
                'scipy>=1.0.0',
                'matplotlib>=2.0.0',
                'corner',
                'h5py>=3.4.0',
                'astropy>=3.0',
                'six',

                ]

test_requirements = ['pytest', ]

setup(
    author="Jeffrey Shafiq Hazboun",
    author_email='jeffrey.hazboun@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Python package for conveniently plotting results from pulsar timing array bayesian analyses. ",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='la_forge',
    name='la_forge',
    packages=find_packages(include=['la_forge']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Hazboun6/la_forge',
    version='1.1.0',
    zip_safe=False,
)
