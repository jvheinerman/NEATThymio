#!/bin/bash

while read line
do
	echo "STARTING $line"
	rsync -rz --progress ./* pi@$line:~/

	git_sha="$(git rev-parse --short HEAD)"
	ssh -X pi@$line './start_one.sh' $line $git_sha &
done < ./bots.txt
wait
