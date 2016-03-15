#!/bin/bash

while read line
do
	echo "STARTING $line"

	if [ ! -f "distances.p" ]; then
		python dist_angle_matrices.py
	fi

	rsync -rzL --progress ./* pi@$line:~/

	git_sha="$(git rev-parse --short HEAD)"
	ssh -X pi@$line './start_one.sh' $1 $line $git_sha &
done < ./bots.txt
wait
