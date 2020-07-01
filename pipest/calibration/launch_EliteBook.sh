#!/bin/bash

symbol="INTC"
date="2019-01-23"
initial_time="36000"
final_time="54000"

python main.py $symbol $date $initial_time $final_time --read
#python3 main.py $symbol $date $initial_time $final_time p
#python3 main.py $symbol $date $initial_time $final_time c
#python3 main.py $symbol $date $initial_time $final_time m

