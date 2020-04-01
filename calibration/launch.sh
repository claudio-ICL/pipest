#!/bin/bash
symbol="INTC"
date="2019-01-04"
t_0="36000"
t_1="36500"

input="symbol=$symbol, date=$date, t_0=$t_0, t_1=$t_1"
a="r"
jobid_r=$(qsub -l walltime=01:00:00 -l select=1:ncpus=1:mem=10gb -N "calibr_main-r" -v "$input, a=$a" wrapper_submission.sh)
echo "submission of 'read' job with jobid: $jobid_r"
a="p"
jobid_p=$(qsub -l walltime=02:00:00 -l select=1:ncpus=5:mem=10gb -N "calibr_main-p" -v "$input, a=$a" -W depend=afterok:$jobid_r wrapper_submission.sh)
echo "submission of 'pre-estimate' with jobid: $jobid_p"
a="c"
jobid_c=$(qsub -l walltime=02:00:00 -l select=1:ncpus=8:mem=20gb -N "calibr_main-c" -J 0-3 -v "$input, a=$a" -W depend=afterok:$jobid_p wrapper_submission.sh)
echo "submission of 'calibrate' job with jobid: $jobid_c"
a="m"
jobid_m=$(qsub -l walltime=01:00:00 -l select=1:ncpus=1:mem=1gb -N "calibr_main-m" -v "$input, a=$a" -W depend=afterok:$jobid_c wrapper_submission.sh)
echo "submission of 'merge' job with jobid: $jobid_m"

