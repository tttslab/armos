#!/bin/bash

for i in `seq 1 168`; do
    echo $i
    num=$((i*1000))
    python eval_corr.py $num
done
