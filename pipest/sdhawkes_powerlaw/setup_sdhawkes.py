from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules=[
        Extension("computation",
            ["computation.pyx"],
            libraries=["m"],
            extra_compile_args = ["-O3","-ffast-math","-march=native","-fopenmp"],
            extra_link_args=['-fopenmp']
            )
]

setup(
  name = "computation",
  cmdclass ={"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()])

ext_modules=[
        Extension(
            "simulation",
            sources=["simulation.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-march=native","-fopenmp"],
            extra_link_args=["-fopenmp"]
            )
        ]

setup(
  name = "simulation",
  cmdclass={"build_ext":build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()])

ext_mod_mle=[
        Extension(
            "mle_estimation",
            sources=["mle_estimation.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-march=native","-fopenmp"],
            extra_link_args=['-fopenmp']
    )
]

setup(
  name = "mle_estimation",
  cmdclass={"build_ext":build_ext},
  ext_modules = ext_mod_mle,
  include_dirs=[numpy.get_include()])

ext_modules=[Extension(
    "goodness_of_fit",
    ["goodness_of_fit.pyx"],
    libraries=["m"]
    )
    ]

ext_mod_gfit=[
        Extension(
            "goodness_of_fit",
            ["goodness_of_fit.pyx"],
           libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-march=native","-fopenmp"],
            extra_link_args=['-fopenmp']
    )
]


setup(
  name = "goodness_of_fit",
  cmdclass={"build_ext":build_ext},
  ext_modules = ext_mod_gfit,
  include_dirs=[numpy.get_include()])


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


setup(
  name = "minimisation_algo",
  ext_modules = cythonize("minimisation_algo.pyx"),
  include_dirs=[numpy.get_include()])

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
