#!/bin/bash
symbol="INTC"
date="2019-01-23"
for phase in "prep" "core" "conclude"
do
  input="symbol=$symbol, date=$date, phase=$phase"
  if [ "$phase" = "core" ]; then
    for quarter in 1 2 3 4; do
      echo "$input, quarte=$quarter"
      echo $quarter
    done
  else
    echo $input
  fi
done

