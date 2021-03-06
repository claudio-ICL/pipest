#!/bin/bash
symbol="INTC"
date="2019-01-23"
timewindow="37800-41400"
br="0.10"
control="0.2"

#action="--read"
#input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action"
#jobid_r=$(qsub -l walltime=05:50:00 -l select=1:ncpus=8:mem=24gb:mpiprocs=1:ompthreads=8 -N "impact-r" -v "$input" wrapper_submission.sh)
#echo "impact-r submitted with jobid: $jobid_r"
#action="--measure"
#input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action, br=$br, control=$control"
#jobid_m=$(qsub -l walltime=05:50:00 -l select=1:ncpus=8:mem=4gb:mpiprocs=1:ompthreads=8 -N "impact-m" -v "$input" -W depend=afterok:$jobid_r wrapper_submission.sh)
#echo "impact-m submitted with jobid: $jobid_m"
action="--panmeasure"
for phase in "prep" "core" "conclude"
do
  input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action, br=$br, control=$control"
  input+=", phase=$phase"
  if [ "$phase" = "core" ]; then
    num_segments="10"
    for segment in 1 2 3 4 5 6 7 8 9 10; do
	  jobid_pm=$(qsub -l walltime=72:00:00  -l select=1:ncpus=32:mem=4gb:mpiprocs=1:ompthreads=32   -N "impact-pm" -J 0-15 -v "$input, segment=$segment, num_segments=$num_segments" -W depend=afterok:$jobid_pm wrapper_submission.sh)
	  echo "impact-pm $phase segment=$segment submitted with jobid: $jobid_pm"
    done
  else
	  jobid_pm=$(qsub -l walltime=72:00:00 -l select=1:ncpus=32:mem=4gb:mpiprocs=1:ompthreads=32 -N "impact-pm" -J 0-15 -v "$input" -W depend=afterok:$jobid_pm wrapper_submission.sh)
	  echo "impact-pm $phase submitted with jobid: $jobid_pm"
  fi
done
action="--collect"
input="symbol=$symbol, date=$date, timewindow=$timewindow, action=$action"
jobid_c=$(qsub -l walltime=00:50:00 -l select=1:ncpus=1:mem=8gb -N "impact-c" -v "$input" -W depend=afterok:$jobid_pm wrapper_submission.sh)
echo "impact-c submitted with jobid: $jobid_c"

