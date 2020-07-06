#!/bin/bash

symbol="INTC"
date="2019-01-23"
time_window="41400-45000"
liquidator_base_rate="0.15"
type_of_liquid="with_the_market"
liquidator_control_type="fraction_of_inventory"
liquidator_control="0.2"

#action="--read"
#python main.py "$symbol" "$date" "$time_window" "$action"
action="--measure"
python main.py "$symbol" "$date" "$time_window" "$action" "$liquidator_base_rate" "$type_of_liquid" "$liquidator_control_type" "$liquidator_control"
