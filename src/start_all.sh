#!/bin/bash

if [ ! -z  $1 ]; then
	BOTS=$1
else
	BOTS=./bots.txt
fi

#sync the clocks
sh ./sync_all.sh $BOTS;

while read line
do
	echo "STARTING $line"

    if [ ! -f "distances.p" ]; then # only if not existing
		python dist_angle_matrices.py
	fi

    git_sha="$(git rev-parse --short HEAD)" # current commit git

    ssh -X pi@$line './start_one.sh' $1 $line $git_sha &  # first command is py file with task
done < $BOTS
wait
