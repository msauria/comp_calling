from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

include_dirs = [numpy.get_include()]

ext_modules=[
    Extension("library", ["library.pyx"], include_dirs=include_dirs, language="c++"),
]

setup(
  name = 'Libraries',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)
