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
import librosa
from nnmnkwii import paramgen as G

logging.basicConfig(filename='log/test.log', level=logging.INFO)

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

windows = [(0, 0, np.array([1.0])),\
           (1, 1, np.array([-0.5, 0.0, 0.5])),\
           (1, 1, np.array([1.0, -2.0, 1.0]))]

device       = 'cuda' if torch.cuda.is_available() else 'cpu'
policy       = models.stacked_BLSTM(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, P_NUM_LAYERS).to(device)
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
        
        mean, var  = policy(inputs, length.tolist())
        
        m   = mean.detach().to('cpu').numpy() 
        v   = var.detach().to('cpu').numpy()
        dm  = librosa.feature.delta(m, width=9, order=1, axis=1)
        ddm = librosa.feature.delta(m, width=9, order=2, axis=1)
        dv  = librosa.feature.delta(v, width=9, order=1, axis=1)
        dv  = 2 * v + dv
        dv  = np.where(dv <= 0, 1e-10, dv)
        ddv = librosa.feature.delta(dv, width=9, order=1, axis=1)
        ddv = 2 * dv + ddv
        ddv = np.where(ddv <= 0, 1e-10, ddv)

        m = np.concatenate((m, dm, ddm), axis=2)
        v = np.concatenate((v, dv, ddv), axis=2)
        action = np.zeros((BATCH_SIZE, length[0], OUT_SIZE))
        for i in range(BATCH_SIZE):
            action[i] = G.mlpg(m[i], v[i], windows)
        action = torch.from_numpy(np.asarray(action, dtype=np.float32)).to(device)
        action = torch.clamp(action, min=0.0, max=1.0)
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
