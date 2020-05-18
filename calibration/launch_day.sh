#!/bin/bash
symbol="INTC"
date="2019-01-15"
t_0=34200
T=57600
dt=3600

mkdir ../models/"$symbol"/"$symbol"_"$date"
while [ $t_0 -lt $T ]
do
    t_1=$(( $t_0 + $dt ))
    input="symbol=$symbol, date=$date, t_0="$t_0", t_1="$t_1""
    echo "$input"
#    a="-r"
#    jobid_r=$(qsub -l walltime=00:20:00 -l select=1:ncpus=1:mem=4gb -N "calibr-r" -v "$input, a=$a" wrapper_submission.sh)
#    echo "submission of 'read' with jobid: $jobid_r"
#    a="-p"
#    jobid_p=$(qsub -l walltime=16:00:00 -l select=1:ncpus=6:mem=40gb -N "calibr-p" -v "$input, a=$a" -W depend=afterok:$jobid_r wrapper_submission.sh)
#echo "submission of 'pre-estimate' with jobid: $jobid_p"
    a="-c"
    ls_jobid_c=()
    for e in 0 1
    do
       jobid_c=$(qsub -l walltime=24:00:00 -v "$input, a=$a, e=$e"  wrapper_submission_mle.sh)
       echo "calibration of component e=$e submitted with jobid: "$jobid_c""
       ls_jobid_c+=("$jobid_c")
    done    
    for e in 2 3 
    do
       jobid_c=$(qsub -l walltime=30:50:00 -v "$input, a=$a, e=$e" wrapper_submission_mle.sh)
       echo "calibration of component e=$e submitted with jobid: "$jobid_c""
       ls_jobid_c+=("$jobid_c")
    done
    a="-m"
   jobid_m=$(qsub -l walltime=01:00:00 -l select=1:ncpus=4:mem=64gb -N "calibr-m" -v "$input, a=$a" -W depend=afterok:${ls_jobid_c[0]}:${ls_jobid_c[1]}:${ls_jobid_c[2]}:${ls_jobid_c[3]} wrapper_submission.sh)
    echo "submission of 'merge' with jobid: $jobid_m"
#    jobid_m=$(qsub -l walltime=01:00:00 -l select=1:ncpus=4:mem=64gb:mpiprocs=1:ompthreads=4 -N "calibr_main-m" -v "$input, a=$a" wrapper_submission.sh)
#    echo "submission of 'merge' with jobid: $jobid_m"
    t_0=$(( $t_0 + $dt ))
done



