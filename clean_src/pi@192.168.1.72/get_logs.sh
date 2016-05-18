#!/bin/bash

while read line
do
	echo "STARTING $line"
	scp -r pi@$line:~/output/NEAT_* ../logs/obs_3_arena/$line/
done < ./bots.txt
wait
