#!/bin/bash
e="0"
jobid_c=$(qsub -l walltime=01:00:00 -l select=1:ncpus=8:mem=30gb:mpiprocs=1:ompthreads=8 -N "calibr_main-c" -v "$input, a=$a, e=$e" -W depend=afterok:$jobid_p wrapper_submission.sh)
echo "calibration of component e=$e submitted with jobid: "$jobid_c""
