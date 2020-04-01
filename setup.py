import os
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<5):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
path_sdhawkes=path_pipest+'/sdhawkes_powerlaw'
path_resources=path_sdhawkes+'/resources'
path_modelling=path_sdhawkes+'/modelling'
path_lobster=path_pipest+'/lobster_for_sdhawkes'
path_lobster_pyscripts=path_lobster+'/py_scripts'


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


#Setup resources
os.chdir(path_resources+'/')
cwd=os.getcwd()
print('\n\nI am performing setup of resources.\nCurrent working directory is '+cwd+'\n')

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


ext_mod_mle=[
        Extension(
            "nonparam_estimation",
            sources=["nonparam_estimation.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-march=native","-fopenmp"],
            extra_link_args=['-fopenmp']
    )
]

setup(
  name = "nonparam_estimation",
  cmdclass={"build_ext":build_ext},
  ext_modules = ext_mod_mle,
  include_dirs=[numpy.get_include()])




ext_modules=[
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
  ext_modules = ext_modules,
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

ext_modules = [
        Extension(
            "minimisation_algo",
            sources=["minimisation_algo.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-march=native","-fopenmp"]
            )
]

setup(
  name = "minimisation_algo",
  cmdclass={"build_ext":build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()])


#Setup modelling
os.chdir(path_modelling+'/')
cwd=os.getcwd()
print('\n\nI am performing setup of modelling.\nCurrent working directory is '+cwd+'\n')

ext_modules = [ 
        Extension( 
            "model",
            sources=["model.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-march=native"]
            )
]

setup(name="model",
    cmdclass={"build_ext":build_ext},    
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()])

ext_modules = [ 
        Extension( 
            "lob_model",
            sources=["lob_model.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-march=native"]
            )
]

setup(name="lob_model",
    cmdclass={"build_ext":build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()])


#Setup lobster
os.chdir(path_lobster_pyscripts+'/')
cwd=os.getcwd()
print('\n\nI am performing setup of lobster.\nCurrent working directory is '+cwd+'\n')

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
