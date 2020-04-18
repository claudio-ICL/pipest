#!/bin/bash
symbol="INTC"
date="2019-01-04"
t_0="36000"
t_1="36200"
input="symbol=$symbol, date=$date, t_0=$t_0, t_1=$t_1"

a="-c"
e="0"
jobid_c=$(qsub -l walltime=01:00:00 -l select=1:ncpus=10:mem=30gb:mpiprocs=1:ompthreads=10 -N "calibr_main-c" -v "$input, a=$a, e=$e" wrapper_submission.sh)
echo "calibration of component e=$e submitted with jobid: "$jobid_c""
