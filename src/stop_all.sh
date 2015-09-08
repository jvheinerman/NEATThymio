#!/bin/sh
while read line
do
	echo "STOPPING $line"
	python ./run_cmd.py $line 54321 --stop &
done <./bots.txt
wait
