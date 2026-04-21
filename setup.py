import os
import shutil
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# If the project uses Intel Fortran (ifx), prefer Intel C/C++ compilers for Cython build.
# Only set them when user has not explicitly provided CC/CXX.
if os.path.exists("Makefile"):
    with open("Makefile", "r", encoding="utf-8") as f:
        makefile_text = f.read()
    if "FC=ifx" in makefile_text or "FC = ifx" in makefile_text:
        if "CC" not in os.environ and shutil.which("icx"):
            os.environ["CC"] = "icx"
        if "CXX" not in os.environ and shutil.which("icpx"):
            os.environ["CXX"] = "icpx"

ext_modules=[Extension('get_ham',['get_ham.pyx'],include_dirs=[np.get_include()])]
setup(
    name= 'get_ham',
    ext_modules=cythonize(ext_modules)
    )
