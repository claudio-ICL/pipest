#!/bin/bash
#PBS -l select=1:ncpus=32:mem=8gb:mpiprocs=1:ompthreads=32
#PBS -N "calibr-c"
cd $PBS_O_WORKDIR
# The variables symbol, date, t_0, t_1, a, and e are imported from the script launch.sh

module load anaconda3/personal
source activate h_impact_env
python main.py $symbol $date $t_0 $t_1 "$a" "-e" "$e"
conda deactivate

