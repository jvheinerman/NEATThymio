#!/bin/sh
while read line
do
	echo "KILLING $line"
	python ./run_cmd.py $line 54321 --kill &
done <./bots.txt
wait
