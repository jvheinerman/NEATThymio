#!/bin/sh
while read line
do
	echo "SHUTTING DOWN $line"
	ssh -n pi@$line sudo shutdown now &
done <./bots.txt
wait