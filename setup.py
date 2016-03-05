#!/usr/bin/env python
from setuptools import setup, find_packages  # This setup relies on setuptools since distutils is insufficient and badly hacked code

version = '0.0.1'
author = 'David-Leon Pohl, Toko Hirono'
author_email = 'pohl@physik.uni-bonn.de,setup.py'

# requirements for core functionality from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='pyBAR_mimosa26_interpreter',
    version=version,
    description='This package can be used to interpred raw data from the Mimosa 26 telescope taken with the readout framework pyBAR. It also contains histogramming functions. The interpretation uses numba JIT to increase the speed.',
    url='https://github.com/SiLab-Bonn/pyBAR_mimosa26_interpreter',
    license='GNU LESSER GENERAL PUBLIC LICENSE Version 2.1',
    long_description='',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    keywords=['mimosa', 'test beam', 'pixel'],
    platforms='any'
)
