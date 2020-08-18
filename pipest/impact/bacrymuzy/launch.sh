#!/bin/bash
symbol="INTC"
date="2019-01-23"
timewindow="37800-41400"
br="0.10"
liquidtype="against_price_move"
controltype="fraction_of_bid_side"
control="0.2"

action="--read"
input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action"
jobid_r=$(qsub -l walltime=05:50:00 -l select=1:ncpus=8:mem=24gb:mpiprocs=1:ompthreads=8 -N "impact-r" -v "$input" wrapper_submission.sh)
echo "impact-r submitted with jobid: $jobid_r"
#action="--measure"
#input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action, br=$br, liquidtype=$liquidtype, controltype=$controltype, control=$control"
#jobid_m=$(qsub -l walltime=05:50:00 -l select=1:ncpus=32:mem=12gb:mpiprocs=1:ompthreads=32 -N "impact-m" -v "$input" -W depend=afterok:$jobid_r wrapper_submission.sh)
#echo "impact-m submitted with jobid: $jobid_m"
#action="--panmeasure"
#input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action, br=$br, control=$control"
#jobid_pm=$(qsub -l walltime=12:50:00 -l select=1:ncpus=32:mem=12gb:mpiprocs=1:ompthreads=32 -N "impact-pm" -J 0-7 -v "$input" -W depend=afterok:$jobid_r wrapper_submission.sh)
#echo "impact-pm submitted with jobid: $jobid_pm"
#action="--collect"
#input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action"
#jobid_c=$(qsub -l walltime=00:50:00 -l select=1:ncpus=1:mem=64gb -N "impact-c" -v "$input" -W depend=afterok:1860004 wrapper_submission.sh)
#echo "impact-c submitted with jobid: $jobid_c"
#
