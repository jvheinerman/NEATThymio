#!/bin/bash
while read line
do
	echo "COPY files TO $line"
	scp ../PycharmProjects/test/algorithmForaging1NN.py  pi@"$line":/home/pi/algorithmForaging.py
	scp ../PycharmProjects/test/parameters1NN.py  pi@"$line":/home/pi/parameters.py
	scp ../PycharmProjects/test/classes1NN.py pi@"$line":/home/pi/classes.py
	scp ../PycharmProjects/test/asebaCommands.aesl pi@"$line":/home/pi/asebaCommands.aesl
	scp config.json pi@"$line":/home/pi/config.json
	echo -e "\r"
done <./bots.txt
wait
