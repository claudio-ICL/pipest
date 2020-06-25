import os
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<4):
    path_pipest=os.path.dirname(path_pipest)
    n+=1
if not os.path.basename(path_pipest)=='pipest':
    print("path_pipest not found. Instead: {}".format(path_pipest))
    raise ValueError("path_pipest not found.")
path_models=path_pipest+'/models'
path_sdhawkes=path_pipest+'/sdhawkes'
path_resources=path_sdhawkes+'/resources/'
path_modelling=path_sdhawkes+'/modelling/'
path_lobster=path_pipest+'/lobster'
path_lobster_data=path_lobster+'/data'
path_lobster_pyscripts=path_lobster+'/py_scripts'
path_tests = path_pipest+'/tests'
path_saved_tests = path_tests+'/saved_tests'
path_perfmeas = path_tests+'/performance_measurements'
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
        Extension("*",
            ["*.pyx"],
            libraries=["m"],
            extra_compile_args = ["-O3","-ffast-math","-fopenmp"],
            extra_link_args=['-fopenmp']
            )
]
setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()])
#Setup modelling
os.chdir(path_modelling+'/')
cwd=os.getcwd()
print('\n\nI am performing setup of modelling.\nCurrent working directory is '+cwd+'\n')
ext_modules = [ 
        Extension( 
            "*",
            sources=["*.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math"]
            )
]
setup(    
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()])
#Setup lobster
os.chdir(path_lobster_pyscripts+'/')
cwd=os.getcwd()
print('\n\nI am performing setup of lobster.\nCurrent working directory is '+cwd+'\n')
ext_modules=[Extension(
    "*",
    sources=["*.pyx"],
    libraries=["m"]
    )
    ]
setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
#Setup performance_measurements
os.chdir(path_perfmeas+'/')
cwd=os.getcwd()
print('\n\nI am performing setup of performance_measurements.\nCurrent working directory is '+cwd+'\n')
ext_modules = [
        Extension(
            "measure_exectime",
            sources=["measure_exectime.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-fopenmp"],
            extra_link_args=['-fopenmp']
            )
]
setup(
  name = "measure_exectime",
  cmdclass={"build_ext":build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()])
