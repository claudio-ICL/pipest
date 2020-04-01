#!/bin/bash

ls -1 ./data/ > ./symbols.txt

for sym in "INTC" "AAPL" "GOOG" "SNAP"
do
	echo "symbol: $sym"
	ls ./data/$sym/*message*10*.csv > ./data/$sym/list_of_csv_messagefiles.txt
	ls ./data/$sym/*orderbook*10.csv > ./data/$sym/list_of_csv_orderbooks.txt
	echo "list of dates for $sym"
	awk -F_ '{print $2}' ./data/$sym/list_of_csv_messagefiles.txt | tee ./data/$sym/dates.txt
	python3 ./py_scripts/produce_dates.py $sym
done


