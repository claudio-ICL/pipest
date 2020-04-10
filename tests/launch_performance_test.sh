#!/bin/bash
p="prange_"
a="e"
jobid_e=$(qsub -l walltime=10:00:00 -l select=1:ncpus=6:mem=30gb -N "perf_test-e" -J 0-3 -v "a=$a, p=$p" wrapper_submission_performance-test.sh)
echo "submission of 'mle_estimation' with jobid: $jobid_e"
a="m"
jobid_m=$(qsub -l walltime=01:50:00 -l select=1:ncpus=1:mem=30gb -N "perf_test-m" -v "a=$a, p=$p" -W depend=afterok:$jobid_e wrapper_submission_performance-test.sh)
echo "submission of 'merge' with jobid: $jobid_m"

