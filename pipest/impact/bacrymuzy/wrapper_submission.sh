#!/bin/bash
module load anaconda3/personal
source activate h_impact_env
cd $PBS_O_WORKDIR
if [ "$action" = "-r" ] || [ "$action" = "--read" ]
then
   python main.py  $symbol $date $timewindow $action
else
   python main.py  $symbol $date $timewindow $action $br $liquidtype $controltype $control
fi
conda deactivate

