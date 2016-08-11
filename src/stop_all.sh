#!/bin/bash

while read line
do
	echo "STOPPING $line"
	{ echo "stop"; sleep 1; } | telnet $line 1337
done < ./bots.txt
wait
