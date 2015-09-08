#!/bin/bash
mkdir received_outputs/"$1"/
while read line
do
	echo "COPY '$1' files FROM $line"
	scp pi@"$line":/home/pi/output/"$1"/"$1"_out.txt received_outputs/"$1"/"$1"_"$line"_out.txt
	scp pi@"$line":/home/pi/output/"$1"/"$1"_sim_debug.log received_outputs/"$1"/"$1"_"$line"_sim_debug.log
	scp pi@"$line":/home/pi/output/"$1"/"$1"_temp.txt received_outputs/"$1"/"$1"_"$line"_temp.txt
	scp pi@"$line":/home/pi/output/"$1"/"$1"_weight_out.txt received_outputs/"$1"/"$1"_"$line"_weight_out.txt
	echo -e "\r"
done <./bots.txt
wait