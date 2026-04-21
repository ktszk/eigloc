import os
import shutil
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# When the Fortran library was built with Intel ifx, prefer the matching Intel
# C/CXX compilers (icx/icpx) for the Cython extension so that OpenMP runtime
# libraries and ABI conventions stay consistent.
# Only override CC/CXX if the user has not set them explicitly.
if os.path.exists("Makefile"):
    with open("Makefile", encoding="utf-8") as f:
        makefile_text = f.read()
    uses_ifx = "FC=ifx" in makefile_text or "FC = ifx" in makefile_text
    if uses_ifx:
        if "CC"  not in os.environ and shutil.which("icx"):
            os.environ["CC"]  = "icx"
        if "CXX" not in os.environ and shutil.which("icpx"):
            os.environ["CXX"] = "icpx"

ext_modules = [
    Extension(
        "get_ham",
        sources=["get_ham.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="get_ham",
    ext_modules=cythonize(ext_modules),
)
