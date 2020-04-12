#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=10gb
#PBS -N perf-meas

num_meas="10"
module load anaconda3/personal
source activate h_impact_env
cd $PBS_O_WORKDIR
python test_perf.py "--loglikelihood" "$num_meas"
conda deactivate
