#!/bin/bash
num_meas="10"
python3 test_perf.py "--loglikelihood" "$num_meas" &
pid_l=$!
echo "measure_loglikelihood launched with pid$pid_l"
wait -n $pid_l
