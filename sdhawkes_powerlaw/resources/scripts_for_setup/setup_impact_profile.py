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

ext_modules = [
        Extension(
            "impact_profile",
            sources=["impact_profile.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3", "-ffast-math", "-march=native","-fopenmp"],
            extra_link_args=['-fopenmp']
            )
]

setup(
  name = "impact_profile",
  cmdclass ={"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()]
  )


