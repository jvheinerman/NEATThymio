#!/bin/bash

while read line
do
	echo "STARTING $line"
	scp -r pi@$line:~/output/NEAT_* ../logs/
done < ./bots.txt
wait
