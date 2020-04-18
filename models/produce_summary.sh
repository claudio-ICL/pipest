#!/bin/bash
symbol="INTC"
date="2019-01-04_36000-39000"
cd ./"$symbol"/"$symbol"_"$date"/readouts/
echo "Calibration on "$symbol"_"$date": summary of readouts" > output_calibration_"$symbol"_"$date".txt
for e in 0 1 2 3
do
	rt=$( grep -E 'run_time|runtime' "$symbol"_"$date"_partial"$e"_readout.txt )
	sed '/Calibration of/'i/"$rt" output_c"$e"_"$date".txt > summary.txt
        sed 's/\/rds\/general\/user\/cb115\/home\///' summary.txt | grep -v "openmp.omp" | grep -v "main.py" | grep -v "^$" > summary_.txt
	sed '/date of run/'i\ '_'  summary_.txt >> output_calibration_"$symbol"_"$date".txt
done
rm summary.txt
rm summary_.txt

 
