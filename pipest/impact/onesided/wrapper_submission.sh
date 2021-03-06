#!/bin/bash
module load anaconda3/personal
source activate h_impact_env
cd $PBS_O_WORKDIR
if [ "$action" = "-m" ] || [ "$action" = "--measure" ]; then
   python main.py  $symbol $date $timewindow $action $br $control
elif [ "$action" = "-pm" ] || [ "$action" = "--panmeasure" ]; then
   python main.py $symbol $date $timewindow $action $phase $segment $num_segments
else
   python main.py  $symbol $date $timewindow $action
fi
conda deactivate

