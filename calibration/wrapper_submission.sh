#!/bin/bash
module load anaconda3/personal
source activate h_impact_env
cd $PBS_O_WORKDIR
if [ "$a" = "-c" ] || [ "$a" = "--calibrate" ]
then 
  python main.py $symbol $date $t_0 $t_1 $a "-e" "$e"
else
  python main.py $symbol $date $t_0 $t_1 $a
fi    
conda deactivate

