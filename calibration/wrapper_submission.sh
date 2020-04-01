#!/bin/bash

echo "wrapper_submission" $symbol $date $t_0 $t_1 $a 

module load anaconda3/personal
source activate h_impact_env

cd $PBS_O_WORKDIR
python main.py $symbol $date $t_0 $t_1 $a  

conda deactivate

