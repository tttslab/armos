#!/bin/bash

cut -f 2 log/train.log | cut -d ' ' -f 2 > log/train_reward_graph.csv
