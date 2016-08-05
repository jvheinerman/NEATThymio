#!/bin/bash

if [ ! -z  $1 ]; then
	BOTS=$1
else
	BOTS=./bots.txt
fi

while read line
do
	echo "STARTING $line"
	scp -r pi@$line:~/output/NEAT_* ../logs/obs_3_arena/$line/
done < $BOTS
wait
