#!/bin/bash
#a="s"
#python3 test_estim_multijob.py "$a" &
#pid_s=$!
#echo "'simulation' launched with pid$pid_s"
#wait -n $pid_s
#a="p"
#python3 test_estim_multijob.py "$a" &
#pid_p=$!
#echo "'non-parametric pre-estimation' launched with pid$pid_p"
#wait -n $pid_p
a='e'
python3 test_estim_multijob.py "$a" &
pid_e=$!
echo "'mle_estimation' launched with pid$pid_e"
wait -n $pid_e
exit 0
a='m'
python3 test_estim_multijob.py "$a" &
pid_m=$!
echo "'merge' launched with pid$pid_m"
wait -n $pid_m
echo "Multijob terminates"
