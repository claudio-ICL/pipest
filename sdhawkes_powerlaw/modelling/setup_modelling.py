from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

setup(name="model",
    ext_modules=cythonize("model.pyx"),
    include_dirs=[numpy.get_include()])

setup(name="lob_model",
        ext_modules=cythonize("lob_model.pyx"),
        include_dirs=[numpy.get_include()])
