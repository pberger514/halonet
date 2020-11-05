from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

compile_args = ['-fopenmp']
link_args = ['-fopenmp']
#compile_args = []
#link_args = []

# Cython module for fast operations
fast_ext = Extension(
    "halonet.tools._fast_tools",
    ["halonet/tools/_fast_tools.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    )

setup(
    name='halonet',
    version=0.1,

    packages=find_packages(),

    ext_modules=cythonize([fast_ext]),

    author="P. Berger, G. Stein",
    author_email="philippe.j.berger@gmail.com",
    description="CNN for mock dark mater halo catalogs"
    )
