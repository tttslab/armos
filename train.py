# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import configparser
import time
import logging
import sys
import os
from models.dpg import DPGAgent
from env.vtl_env import VTLEnv


model_name = sys.argv[1]


### Hyperparameters

if model_name == "dpg_pn":
    # TODO: 違うモデルでも整合性が取れるようにする
    logging.basicConfig(filename=os.path.join('log', model_name, 'train.log'), level=logging.INFO)
    conf = configparser.SafeConfigParser()
    conf.read(os.path.join('config', model_name, "config.ini"))
    IN_SIZE             = int(conf.get('main', 'in_size'))
    OUT_SIZE            = int(conf.get('main', 'out_size'))
    P_HIDDEN_SIZE       = int(conf.get('actor', 'hidden_size'))
    P_NUM_LAYERS        = int(conf.get('actor', 'num_layers'))
    DELTA               = float(conf.get('actor', 'delta'))
    SIGMA_INIT          = float(conf.get('actor', 'sigma_init'))
    ADJUST_STEP         = int(conf.get('actor', 'adjust_step'))
    P_LEARNING_RATE     = float(conf.get('actor', 'learning_rate'))
    Q_HIDDEN_SIZE       = int(conf.get('critic', 'hidden_size'))
    Q_LEARNING_RATE     = float(conf.get('critic', 'learning_rate'))
    BATCH_SIZE          = int(conf.get('main', 'batch_size'))
    NUM_PARAL           = int(conf.get('main', 'num_paral'))
    AUDIO_SEGMENT       = int(conf.get('main', 'audio_segment'))
    FRAME_RATE_HZ       = int(conf.get('main', 'frameRate_Hz'))

    agent = DPGAgent(IN_SIZE,
                    OUT_SIZE,
                    P_HIDDEN_SIZE,
                    P_NUM_LAYERS,
                    DELTA,
                    SIGMA_INIT,
                    ADJUST_STEP,
                    P_LEARNING_RATE,
                    Q_HIDDEN_SIZE,
                    Q_LEARNING_RATE,
                    BATCH_SIZE)
else:
    raise ValueError('The first arg should be avalue of "dpg"')

env = VTLEnv(
    IN_SIZE=IN_SIZE,
    BATCH_SIZE=BATCH_SIZE,
    AUDIO_SEGMENT=AUDIO_SEGMENT,
    FRAME_RATE_HZ=FRAME_RATE_HZ,
    NUM_PARAL=NUM_PARAL)
state = env.reset()

for iteration in range(1000): #0000):
    start = time.time()

    action = agent.act(state)
    state, reward, _, _ = env.step(action)

    loss, sigma_init = agent.update(reward)

    env.render(iteration, loss, sigma_init, time.time()-start)

