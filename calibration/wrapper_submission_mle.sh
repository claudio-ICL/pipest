#!/bin/bash
#PBS -l select=1:ncpus=24:mem=16gb:mpiprocs=1:ompthreads=24
#PBS -N "calibr-c"
cd $PBS_O_WORKDIR
# The variables symbol, date, t_0, t_1, a, and e are imported from the script launch.sh

module load anaconda3/personal
source activate h_impact_env
python main.py $symbol $date $t_0 $t_1 "$a" "-e" "$e"
conda deactivate

