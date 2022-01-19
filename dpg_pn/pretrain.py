import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
import utils
import configparser
import time
import logging
import librosa
import random
import copy

logging.basicConfig(filename='log/pretrain.log', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')

### Hyperparameters
conf = configparser.SafeConfigParser()
conf.read("config.ini")
IN_SIZE             = int(conf.get('main', 'in_size'))
OUT_SIZE            = int(conf.get('main', 'out_size'))
P_HIDDEN_SIZE       = int(conf.get('actor', 'hidden_size'))
P_NUM_LAYERS        = int(conf.get('actor', 'num_layers'))
Q_HIDDEN_SIZE       = int(conf.get('critic', 'hidden_size'))
BATCH_SIZE          = int(conf.get('main', 'batch_size'))
SAMPLING_RATE       = int(conf.get('main', 'sampling_rate'))
NUM_PARAL           = int(conf.get('main', 'num_paral'))
AUDIO_SEGMENT       = int(conf.get('main', 'audio_segment'))
FRAMERATE_HZ        = int(conf.get('main', 'frameRate_Hz'))

### Condition Setting
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
# policy       = models.stacked_BLSTM(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, P_NUM_LAYERS, 0).to(device)
# policy       = models.stacked_Attention(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, 0).to(device)
policy       = models.stacked_BLSTM_Attention(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, P_NUM_LAYERS, 0).to(device)
loss_fun     = nn.MSELoss()
p_optim      = torch.optim.Adam(policy.parameters(), lr=1e-3)
train_loader = utils.Batch_generator('training', BATCH_SIZE)

for iteration in range(100):
    policy.train()
    start = time.time()

    ### Assume the duration is multiple of 10ms.
    ### For example, if the time length of input feature is 101, this sound is from 1000ms to 1009ms,
    ### but I regard this as 1000ms and ignore the last feature.
    inputs, length, _ = next(train_loader)

    feats = np.zeros((BATCH_SIZE, AUDIO_SEGMENT, IN_SIZE))
    for i in range(BATCH_SIZE):
        s_pos    = np.random.randint(length[i]-AUDIO_SEGMENT+1)
        feats[i] = inputs[i, s_pos:s_pos+AUDIO_SEGMENT,:]
    inputs = np.asarray(feats, dtype=np.float32)
    inputs            = torch.from_numpy(inputs).to(device)
    length            = np.asarray(length, dtype=np.int32) - 1
    length            = np.where(length > AUDIO_SEGMENT, AUDIO_SEGMENT, length)
    dur               = 1.0*length / FRAMERATE_HZ ### dimension is sec.
    target = torch.ones(BATCH_SIZE, AUDIO_SEGMENT, OUT_SIZE).to(device) * 0.5

    ### Get action
    action = policy(inputs, length.tolist(), False)
    loss   = F.mse_loss(action, target)
    p_optim.zero_grad()
    loss.backward()
    p_optim.step()

    logging.info('iter: {}\tloss: {}\telapse: {:.02f}'.format(iteration+1,\
                                                              loss.item(),\
                                                              time.time()-start))
torch.save(policy.state_dict(),  './exp/pretrain.model')
