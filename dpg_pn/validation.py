import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
import utils
import configparser
import sys
import time
import logging

logging.basicConfig(filename='log/valid.log', level=logging.INFO)

conf = configparser.SafeConfigParser()
conf.read("config.ini")
IN_SIZE             = int(conf.get('main', 'in_size'))
OUT_SIZE            = int(conf.get('main', 'out_size'))
P_HIDDEN_SIZE       = int(conf.get('actor', 'hidden_size'))
P_NUM_LAYERS        = int(conf.get('actor', 'num_layers'))
Q_HIDDEN_SIZE       = int(conf.get('critic', 'hidden_size'))
#BATCH_SIZE          = int(conf.get('main', 'batch_size'))
SAMPLING_RATE       = int(conf.get('main', 'sampling_rate'))
frameRate_Hz        = int(conf.get('main', 'frameRate_Hz'))
FEAT_TYPE           = conf.get('main', 'feat_type')
BATCH_SIZE = int(sys.argv[3])
NUM_PARAL  = int(sys.argv[4])

device       = 'cuda' if torch.cuda.is_available() else 'cpu'
policy       = models.stacked_BLSTM(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, P_NUM_LAYERS, 0).to(device)
data_loader = utils.Batch_generator(sys.argv[2], BATCH_SIZE)

num = sys.argv[1]
policy.load_state_dict(torch.load('exp/p' + num + '.model'))

reward_mean = 0
total_frame = 0
e = 1
iteration = 0
with torch.no_grad():
    policy.eval()
    start = time.time()
    while e < 2:
        feats, length, e = next(data_loader)
        inputs           = np.asarray(feats, dtype=np.float32)
        inputs           = torch.from_numpy(inputs).to(device)
        length           = np.asarray(length, dtype=np.int32)
        dur              = 1.0 * length / frameRate_Hz
        
        action   = policy(inputs, length.tolist(), False)
        r_calc_t = utils.trans_param(action.detach())[:, :, :24]
        r_calc_g = utils.trans_param(action.detach())[:, :, -6:]
        r_calc = utils.calc_reward(inputs, r_calc_t,    r_calc_g, length, dur, NUM_PARAL)
        
        for i in range(BATCH_SIZE):
            reward_mean += F.mse_loss(torch.from_numpy(r_calc[i][:length[i]]).to(device), inputs[i, :length[i]], reduction='none').mean(dim=1).sum()
        total_frame += length.sum()

        iteration += 1
        if iteration % 10 == 0:
            logging.info('{} iter passed'.format(iteration))
    logging.info(reward_mean/total_frame)
    logging.info(time.time()-start)
