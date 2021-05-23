#!/bin/bash

for keyword in bed bird cat dog down eight five four go happy \
house left marvin nine no off on one right seven sheila six stop \
three tree two up wow yes zero; do
    mkdir -p remsil_speech_commands_v0.01/$keyword
    for wav in `ls speech_commands_v0.01/$keyword/*.wav`; do
	bname=`basename $wav`
	sox $wav remsil_speech_commands_v0.01/$keyword/$bname silence 1 0.1 1% reverse silence 1 0.1 1% reverse
    done
done
find remsil_speech_commands_v0.01/ -name "*.wav" -size 44c -exec rm {} \;
