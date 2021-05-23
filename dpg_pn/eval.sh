#!/bin/bash

for i in `seq 1 51`; do
    echo $i
    num=$((i*1000))
    python eval.py $num
done
