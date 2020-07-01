#!/bin/bash
symbol="INTC"
date="2019-01-23"
timewindow="41400-45000"
br="0.15"
liquidtype="with_the_market"
controltype="fraction_of_inventory"
control="0.2"

action="--read"
input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action"
jobid_r=$(qsub -l walltime=00:50:00 -l select=1:ncpus=8:mem=4gb:mpiprocs=1:ompthreads=8 -N "impact-r" -v "$input" wrapper_submission.sh)
echo "impact-r submitted with jobid: $jobid_r"
action="--measure"
input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action, br=$br, liquidtype=$liquidtype, controltype=$controltype, control=$control"
jobid_m=$(qsub -l walltime=00:50:00 -l select=1:ncpus=8:mem=4gb:mpiprocs=1:ompthreads=8 -N "impact-m" -v "$input" -W depend=afterok:$jobid_r wrapper_submission.sh)
echo "impact-m submitted with jobid: $jobid_m"

