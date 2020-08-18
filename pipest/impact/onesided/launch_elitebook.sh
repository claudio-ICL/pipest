#!/bin/bash

symbol="INTC"
date="2019-01-23"
time_window="37800-41400"
liquidator_base_rate="0.20"
liquidator_control="0.2"

#action="--read"
#python main.py "$symbol" "$date" "$time_window" "$action"
action="--measure"
python main.py "$symbol" "$date" "$time_window" "$action" "$liquidator_base_rate" "$liquidator_control"
