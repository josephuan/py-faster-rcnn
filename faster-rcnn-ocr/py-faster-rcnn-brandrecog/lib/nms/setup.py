from distutils.core import setup
from Cython.Build import cythonize
setup(name = 'gpu_nms app',
      ext_modules = cythonize("gpu_nms.pyx"))