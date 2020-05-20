#!/bin/bash
symbol="INTC"
date="2019-01-31_43200-46800"
cd ./"$symbol"/"$symbol"_"$date"/readouts/
echo "Calibration on "$symbol"_"$date": summary of readouts" > output_calibration_"$symbol"_"$date".txt
for e in 0 1 2 3
do
	cal_info_1=$( grep 'Calibration is being' "$symbol"_"$date"_partial"$e"_readout.txt )
	cal_info_2=$( grep 'posix' "$symbol"_"$date"_partial"$e"_readout.txt )
	rt=$( grep 'estimate_hawkes_power_partial, event_type=' "$symbol"_"$date"_partial"$e"_readout.txt )
	#echo $cal_info_1
	#echo $cal_info_2
	#echo $rt
	sed '/stdout is being/'i\ "$cal_info_1" output_c"$e"_"$date".txt | sed '/stdout is bein/'i\ "$cal_info_2" | sed '/Calibration of/'i\ "$rt" > summary.txt
        sed 's/\/rds\/general\/user\/cb115\/home\///' summary.txt | grep -v "openmp.omp" | grep -v "main.py" | grep -v "Initialising" | grep -v "^$" > summary_.txt
	sed '/date of run/'i\ '_'  summary_.txt >> output_calibration_"$symbol"_"$date".txt
done
rm summary.txt
rm summary_.txt

 
