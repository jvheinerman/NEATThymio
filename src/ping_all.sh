#!/bin/sh
while read line
do
	echo "\nPINGING $line:"
	ping -c3 "$line"
done <./bots.txt
wait
