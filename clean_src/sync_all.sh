#!/bin/bash

while read line
do
	echo "SYNC CLOCK AND FILES FOR: $line"
	rsync -rzL --exclude 'logs' ./* pi@$line:~/ #copy files only that changed
    ssh -n pi@$line 'sudo rdate -ncv 0.nl.pool.ntp.org'
done < ./bots.txt
wait
