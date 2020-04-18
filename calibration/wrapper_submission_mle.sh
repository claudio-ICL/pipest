#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=16:mem=30gb:mpiprocs=1:ompthreads=16
#PBS -N "calibr-c"
cd $PBS_O_WORKDIR
#symbol="INTC"
#date="2019-01-04"
#t_0="36000"
#t_1="36200"
#input="symbol=$symbol, date=$date, t_0=$t_0, t_1=$t_1"
a="-c"
e="0"
module load anaconda3/personal
source activate h_impact_env
python main.py $symbol $date $t_0 $t_1 "$a" "-e" "$e"
conda deactivate

