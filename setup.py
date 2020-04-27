#!/usr/bin/env python
from setuptools import setup, find_packages  # This setup relies on setuptools since distutils is insufficient and badly hacked code

author = 'Yannick Dieter, Toko Hirono, Jens Janssen, David-Leon Pohl, Pascal Wolf'
author_email = 'dieter@physik.uni-bonn.de, hirono@physik.uni-bonn.de, janssen@physik.uni-bonn.de, pohl@physik.uni-bonn.de, wolf@physik.uni-bonn.de'

with open('VERSION') as version_file:
    version = version_file.read().strip()

# requirements for core functionality from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='pymosa_mimosa26_interpreter',
    version=version,
    description='Interpreter for Mimosa26 raw data recorded with pymosa DAQ',
    url='https://github.com/SiLab-Bonn/pymosa_mimosa26_interpreter',
    license='BSD 3-Clause License',
    long_description='',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    keywords=['mimosa26', 'test-beam', 'pixel', 'telescope'],
    python_requires='>=2.7',
    platforms='any'
)
