#!/bin/bash
date="2020-04-12_0738"
cd ./perf-test_"$date"/readouts/
for e in 0 1 2 3
do
	echo "EVENT TYPE $e" > summary.txt
	for type in "plain" "prange" "pool"
	do
		rt=$(grep -E 'runtime|run_time' "$type"_test_model_"$date"_mle_readout_partial"$e".txt)
		sed '/Estimation of/'i\ "$rt" "$type"_output_e"$e"_"$date".txt >> summary.txt
	done
	grep -v "switch=" summary.txt | grep -v "main() end" | grep -v "\$python test_" | grep -v "openmp." | grep -v '^$' > summary_.txt
	sed 's/\/rds\/general\/user\/cb115\/home\///' summary_.txt > summary_e"$e".txt
done
rm summary.txt
cd ..
