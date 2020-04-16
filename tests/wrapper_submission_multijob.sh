#!/bin/bash

module load anaconda3/personal
source activate h_impact_env

cd $PBS_O_WORKDIR
if [ "$a" = "-e" ] || [ "$a" = "--estimate" ] 
then
   python test_estim_multijob.py "$a" "$p"
else
   python test_estim_multijob.py "$a"
fi
conda deactivate
