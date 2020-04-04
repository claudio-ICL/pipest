#!/bin/bash
a="s"
jobid_s=$(qsub -l walltime=00:10:00 -l select=1:ncpus=1:mem=1gb -N "multijob_test-s" -v "a=$a" wrapper_submission_multijob.sh)
echo "submission of 'simulation' with jobid: $jobid_s"
a="p"
jobid_p=$(qsub -l walltime=00:50:00 -l select=1:ncpus=4:mem=20gb -N "multijob_test-p" -v "a=$a" -W depend=afterok:$jobid_s wrapper_submission_multijob.sh)
echo "submission of 'pre-estimate' with jobid: $jobid_p"
a="e"
jobid_e=$(qsub -l walltime=01:00:00 -l select=1:ncpus=6:mem=20gb -N "multijob_test-e" -J 0-3 -v "a=$a" -W depend=afterok:$jobid_p wrapper_submission_multijob.sh)
echo "submission of 'mle_estimation' with jobid: $jobid_e"
a="m"
jobid_m=$(qsub -l walltime=00:20:00 -l select=1:ncpus=1:mem=10gb -N "multijob_test-m" -v "a=$a" -W depend=afterok:$jobid_e wrapper_submission_multijob.sh)
echo "submission of 'merge' with jobid: $jobid_m"

