#!/bin/sh
python accept_output_file.py &
while read line
do
	echo "STARTING $line"
	if [ -n "$1" ]
	then
		python ./run_cmd.py $line 54321 --start --debug &
	else
		python ./run_cmd.py $line 54321 --start &
	fi
done < ./bots.txt
wait
