#!/bin/bash
symbol="INTC"
date="2019-01-04"
t_0="36000"
t_1="37000"

input="symbol=$symbol, date=$date, t_0=$t_0, t_1=$t_1"
a="-r"
jobid_r=$(qsub -l walltime=00:20:00 -l select=1:ncpus=1:mem=4gb -N "calibr_main-r" -v "$input, a=$a" wrapper_submission.sh)
echo "submission of 'read' with jobid: $jobid_r"
a="-p"
jobid_p=$(qsub -l walltime=00:50:00 -l select=1:ncpus=6:mem=30gb -N "calibr_main-p" -v "$input, a=$a" -W depend=afterok:$jobid_r wrapper_submission.sh)
echo "submission of 'pre-estimate' with jobid: $jobid_p"
a="-c"
ls_jobid_c=()
for e in 0 1 2 3
do
    jobid_c=$(qsub -l walltime=00:20:00 -N "calibr_main-c" -v "$input, a=$a, e=$e" -W depend=afterok:$jobid_p wrapper_submission_mle.sh)
    echo "calibration of component e=$e submitted with jobid: "$jobid_c""
    ls_jobid_c+=("$jobid_c")
done    
a="-m"
jobid_m=$(qsub -l walltime=00:50:00 -l select=1:ncpus=1:mem=30gb -N "calibr_main-m" -v "$input, a=$a" -W depend=afterok:${ls_jobid_c[0]}:${ls_jobid_c[1]}:${ls_jobid_c[2]}:${ls_jobid_c[3]} wrapper_submission.sh)
echo "submission of 'merge' with jobid: $jobid_m"

