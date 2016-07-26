from distutils.core import setup
from Cython.Build import cythonize
setup(name = 'bbox app',
      ext_modules = cythonize("bbox.pyx"))