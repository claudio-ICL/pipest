#!/bin/bash

symbol="INTC"
date="2019-01-23"
timewindow="37800-41400"
liquidator_base_rate="0.10"
type_of_liquid="with_price_move"
liquidator_control_type="fraction_of_inventory"
liquidator_control="0.2"

#action="--read"
#python main.py "$symbol" "$date" "$timewindow" "$action"
action="--measure"
python main.py "$symbol" "$date" "$timewindow" "$action" "$liquidator_base_rate" "$type_of_liquid" "$liquidator_control_type" "$liquidator_control"
