#!/bin/bash

symbol="INTC"
date="2019-01-04"
initial_time="36000"
final_time="36400"

python3 main_EliteBook.py $symbol $date $initial_time $final_time r
python3 main_EliteBook.py $symbol $date $initial_time $final_time c
python3 main_EliteBook.py $symbol $date $initial_time $final_time m

