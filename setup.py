from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules=[Extension('get_ham',['get_ham.pyx'],include_dirs=[np.get_include()])]
setup(
    name= 'get_ham',
    ext_modules=cythonize(ext_modules)
    )
