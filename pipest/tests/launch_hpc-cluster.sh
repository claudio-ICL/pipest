#!/bin/bash
a="-s"
jobid_s=$(qsub -l walltime=00:20:00 -l select=1:ncpus=2:mem=5gb -N "multijob_test-s" -v "a=$a" wrapper_submission_multijob.sh)
echo "submission of 'simulation' with jobid: $jobid_s"
# a="-p"
# jobid_p=$(qsub -l walltime=04:00:00 -l select=1:ncpus=4:mem=30gb -N "multijob_test-p" -v "a=$a" -W depend=afterok:$jobid_s wrapper_submission_multijob.sh)
# echo "submission of 'pre-estimate' with jobid: $jobid_p"
# a="-e"
# jobid_e=$(qsub -l walltime=02:00:00 -l select=1:ncpus=8:mem=30gb:mpiprocs=1:ompthreads=8 -N "multijob_test-e" -J 0-3 -v "a=$a" -W depend=afterok:$jobid_p wrapper_submission_multijob.sh)
# echo "submission of 'mle_estimation' with jobid: $jobid_e"
# a="-m"
# jobid_m=$(qsub -l walltime=00:50:00 -l select=1:ncpus=1:mem=30gb -N "multijob_test-m" -v "a=$a" -W depend=afterok:$jobid_e wrapper_submission_multijob.sh)
# echo "submission of 'merge' with jobid: $jobid_m"

