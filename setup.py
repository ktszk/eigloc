from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension('get_ham',['get_ham.pyx'])]
setup(
    name= 'get_ham',
    cmdclass={'build_ext':build_ext},
    ext_modules=ext_modules
    )
