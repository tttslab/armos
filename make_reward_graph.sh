#!/bin/bash

cut -f 2 log/$1/train.log | cut -d ' ' -f 2 > log/$1/train_reward_graph.csv
