from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules=[Extension(
    "prepare_from_lobster",
    sources=["prepare_from_lobster.pyx"],
    libraries=["m"]
    )
    ]

setup(name="prepare_from_lobster",
      cmdclass={"build_ext":build_ext},
      ext_modules = ext_modules,
      include_dirs=[numpy.get_include()]
      )
