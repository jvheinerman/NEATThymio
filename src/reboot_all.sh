#!/bin/sh
while read line
do
	echo "REBOOTING $line"
	ssh -n pi@$line sudo reboot &
done <./bots.txt
wait
