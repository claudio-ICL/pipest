#!/bin/bash
symbol="INTC"
date="2019-01-23"
timewindow="45000-48600"
br="0.10"
liquidtype="with_the_market"
controltype="fraction_of_inventory"
control="0.2"

action="--read"
input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action"
jobid_r=$(qsub -l walltime=05:50:00 -l select=1:ncpus=8:mem=24gb:mpiprocs=1:ompthreads=8 -N "impact-r" -v "$input" wrapper_submission.sh)
echo "impact-r submitted with jobid: $jobid_r"
#action="--measure"
#input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action, br=$br, liquidtype=$liquidtype, controltype=$controltype, control=$control"
#jobid_m=$(qsub -l walltime=05:50:00 -l select=1:ncpus=8:mem=4gb:mpiprocs=1:ompthreads=8 -N "impact-m" -v "$input" -W depend=afterok:$jobid_r wrapper_submission.sh)
#echo "impact-m submitted with jobid: $jobid_m"
action="--panmeasure"
input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action, br=$br, control=$control"
jobid_pm=$(qsub -l walltime=17:50:00 -l select=1:ncpus=8:mem=4gb:mpiprocs=1:ompthreads=8 -N "impact-pm" -J 0-5 -v "$input" -W depend=afterok:$jobid_r wrapper_submission.sh)
echo "impact-pm submitted with jobid: $jobid_pm"
action="--collect"
input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action"
jobid_c=$(qsub -l walltime=00:50:00 -l select=1:ncpus=1:mem=8gb -N "impact-c" -v "$input" -W depend=afterok:$jobid_pm wrapper_submission.sh)
echo "impact-c submitted with jobid: $jobid_c"

