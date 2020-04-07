#!/bin/bash

module load anaconda3/personal
source activate h_impact_env

cd $PBS_O_WORKDIR
python test_estim_multijob.py "$a" ""

conda deactivate
