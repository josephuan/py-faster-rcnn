from distutils.core import setup
from Cython.Build import cythonize
setup(name = '_mask app',
      ext_modules = cythonize("_mask.pyx"))