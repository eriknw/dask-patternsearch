#!/usr/bin/env python

import os.path
from setuptools import setup
import dask_patternsearch

install_requires = [
    'dask',
    'distributed',  # TODO: needs develop version
    'toolz',
    'numpy',
]

setup(
    name='dask-patternsearch',
    version=dask_patternsearch.__version__,
    description='Scalable pattern search optimization with dask',
    url='https://github.com/eriknw/dask-patternsearch',
    maintainer='Erik Welch',
    maintainer_email='erik.n.welch@gmail.com',
    license='BSD',
    keywords=['parallel pydata dask optimize optimization'],
    packages=['dask_patternsearch'],
    long_description=(open('README.rst').read()
                      if os.path.exists('README.rst')
                      else ''),
    install_requires=install_requires,
    zip_safe=False,
)
