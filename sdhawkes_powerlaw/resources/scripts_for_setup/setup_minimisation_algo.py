from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

# Run in the console (under Anaconda version): python setup.py build_ext --inplace

# setup(
#     ext_modules=cythonize("hybrid_hawkes_exp_likelihood.pyx"),
#     include_dirs=[numpy.get_include()]
# )

setup(
  name = "minimisation_algo",
  ext_modules = cythonize("minimisation_algo.pyx"),
  include_dirs=[numpy.get_include()])

