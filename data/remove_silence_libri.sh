#!/bin/bash

mkdir -p remsil_LibriSpeech
mkdir -p tmp

name=1
# FIXME: Maybe something's wrong.
for sound in `ls ./LibriSpeech/train-clean-100/*/*/*.flac`; do
    sox $sound tmp/out.wav silence 1 0.2 1% 1 0.2 1% : newfile : restart

    for splitted in `ls tmp/*.wav`; do
	dur=`soxi $splitted | grep 'Duration' | cut -d ':' -f 4`
	min=${dur:0:2}
	sec=${dur:3:2}
	if [ -n "$dur" ]; then
	    if [ $sec -gt 30 ]; then
		mv $splitted remsil_LibriSpeech/${name}.wav
		name=$((name+1))
	    elif [ $min -gt 0 ]; then
		mv $splitted remsil_LibriSpeech/${name}.wav
		name=$((name+1))
	    fi
	fi
    done
    rm -r tmp
    mkdir -p tmp
    if [ $((name%100)) -eq 0 ]; then
	echo $name
    fi
done

rm -r tmp
