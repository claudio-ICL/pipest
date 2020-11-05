#!/bin/bash

symbol="INTC"
date="2019-01-23"
timewindow="37800-41400"
br="0.20"
control="0.2"

#action="--read"
#python main.py "$symbol" "$date" "$time_window" "$action"
action="--measure"
for phase in "prep" "core" "conclude"; do
  if [ "$phase" = "core" ]; then
    for quarter in 1 2 3 4; do
      python main.py "$symbol" "$date" "$timewindow" "$action" "$br" "$control" "$phase" "$quarter"
    done
  else
      python main.py "$symbol" "$date" "$timewindow" "$action" "$br" "$control" "$phase" 
  fi
done
#action="--collect"
#python main.py $symbol $date $timewindow $action
