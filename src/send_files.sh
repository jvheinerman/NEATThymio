#!/bin/bash
while read line
do
	echo "COPY files TO $line"
	scp Algorithm/algorithmForaging.py  pi@"$line":/home/pi/algorithmForaging.py
	scp Algorithm/parameters.py  pi@"$line":/home/pi/parameters.py
	scp Algorithm/classes.py pi@"$line":/home/pi/classes.py
	scp Algorithm/asebaCommands.aesl pi@"$line":/home/pi/asebaCommands.aesl
	scp config.json pi@"$line":/home/pi/config.json
	echo -e "\r"
done <./bots.txt
wait
