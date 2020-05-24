#!/usr/bin/env python
import os
import sys
import glob
path_pipest = os.path.abspath('./pipest/')
if not os.path.basename(path_pipest)=='pipest':
    print("path_pipest not found. Instead: {}".format(path_pipest))
    raise ValueError("path_pipest not found.")
path_models=path_pipest+'/models'
path_sdhawkes=path_pipest+'/sdhawkes_powerlaw'
path_resources=path_sdhawkes+'/resources'
path_modelling=path_sdhawkes+'/modelling'
path_lobster=path_pipest+'/lobster_for_sdhawkes'
path_lobster_data=path_lobster+'/data'
path_lobster_pyscripts=path_lobster+'/py_scripts'
path_tests = path_pipest+'/tests'
path_saved_tests = path_tests+'/saved_tests'
path_perfmeas = path_tests+'/performance_measurements'
import setuptools 
import numpy
ss=[]
extension_list=[]
for domain in [path_modelling+'/', path_resources+'/', path_lobster_pyscripts+'/']:
    os.chdir(domain)
    [extension_list.append(
        setuptools.Extension("pipest.{}".format(name[:-2]),
            sources=[domain+"{}".format(name)],
            libraries=["m"],
            extra_compile_args=["-O3","-ffast-math","-fopenmp"],
            extra_link_args=['-fopenmp']
            )
        ) for name in glob.glob("*.c")]
    [ss.append(domain+"{}".format(name)) for name in glob.glob("*.c")]
os.chdir(path_pipest)
print(extension_list)
print("\nList of paths:\n")
print(ss)
setuptools.setup(
        ext_modules=extension_list,
        include_dirs=[numpy.get_include()],
        name="pipest",
       packages=setuptools.find_packages(),
)
