#!/bin/bash

while read line
do
	echo "STARTING $line"

    if [ ! -f "distances.p" ]; then # only if not existing
		python dist_angle_matrices.py
	fi

    rsync -rzL --progress ./* pi@$line:~/ #copy files only that changed

    git_sha="$(git rev-parse --short HEAD)" # current commit git
    ssh -X pi@$line './start_one.sh' $1 $line $git_sha &  # first command is py file with task
done < ./bots.txt
wait
