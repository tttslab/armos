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
from nnmnkwii import paramgen as G

logging.basicConfig(filename='log/train.log', level=logging.INFO)

### Hyperparameters
conf = configparser.SafeConfigParser()
conf.read("config.ini")
IN_SIZE             = int(conf.get('main', 'in_size'))
OUT_SIZE            = int(conf.get('main', 'out_size'))
P_HIDDEN_SIZE       = int(conf.get('actor', 'hidden_size'))
P_NUM_LAYERS        = int(conf.get('actor', 'num_layers'))
P_LEARNING_RATE     = float(conf.get('actor', 'learning_rate'))
Q_HIDDEN_SIZE       = int(conf.get('critic', 'hidden_size'))
Q_LEARNING_RATE     = float(conf.get('critic', 'learning_rate'))
BATCH_SIZE          = int(conf.get('main', 'batch_size'))
NUM_PARAL           = int(conf.get('main', 'num_paral'))
AUDIO_SEGMENT       = int(conf.get('main', 'audio_segment'))
frameRate_Hz        = int(conf.get('main', 'frameRate_Hz'))

### Condition Setting
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
policy       = models.stacked_BLSTM(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, P_NUM_LAYERS).to(device)
q_func       = models.Qfunction(IN_SIZE, OUT_SIZE, Q_HIDDEN_SIZE).to(device)
loss_fun     = nn.MSELoss(reduction='none')
p_optim      = torch.optim.SGD(policy.parameters(), lr=P_LEARNING_RATE)
q_optim      = torch.optim.Adam(q_func.parameters(), lr=Q_LEARNING_RATE)
train_loader = utils.Batch_generator('training', BATCH_SIZE)
windows = [(0, 0, np.array([1.0])),\
           (1, 1, np.array([-0.5, 0.0, 0.5])),\
           (1, 1, np.array([1.0, -2.0, 1.0]))]

policy.load_state_dict(torch.load('exp/pretrain.model'))
#p_optim.load_state_dict(torch.load('exp/p_optim.state'))
#q_func.load_state_dict(torch.load('exp/q1000.model'))
#q_optim.load_state_dict(torch.load('exp/q_optim.state'))

for iteration in range(10000000):
    policy.train()
    q_func.train()
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
    inputs = torch.from_numpy(inputs).to(device)
    length = np.asarray(length, dtype=np.int32) - 1
    length = np.where(length > AUDIO_SEGMENT, AUDIO_SEGMENT, length)
    dur    = 1.0 * length / frameRate_Hz ### dimension is sec.

    ### Get action
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
    gauss_dist = torch.distributions.normal.Normal(mean, var)
    log_prob   = gauss_dist.log_prob(action).sum(dim=-1).sum(dim=-1)

    ### Store parameters
    tractParams   = utils.trans_param(action)[:, :, :24]
    glottisParams = utils.trans_param(action)[:, :, -6:]

    ### Reward calculation
    reward = utils.calc_reward(inputs, tractParams, glottisParams, length, dur, NUM_PARAL)
    reward_mean = 0
    for i in range(BATCH_SIZE):
        reward_mean += F.mse_loss(torch.from_numpy(reward[i][:length[i]]).to(device), inputs[i, :length[i]], reduction='none').mean(dim=1).sum()
    reward_mean /= length.sum()
    
    ### update Q-function
    q_value = q_func(action, length.tolist())
    loss    = 0
    for i in range(BATCH_SIZE):
        loss += loss_fun(q_value[i,:length[i]], torch.from_numpy(reward[i][:length[i]]).to(device)).mean(dim=1).sum()
    loss /= length.sum()
    q_optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_func.parameters(), 10.0)
    q_optim.step()

    ### update policy for critic
    reconst = q_func(action, length.tolist())
    p_loss  = torch.zeros(BATCH_SIZE)
    for i in range(BATCH_SIZE):
        p_loss[i] = loss_fun(reconst[i, :length[i]], inputs[i, :length[i]]).mean(dim=1).sum()/length[i]
    if (iteration)%10 == 0:
        x = 1+1
    p_loss = log_prob * p_loss.to(device)
    p_loss = p_loss.mean()
    p_optim.zero_grad()
    p_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    p_optim.step()

    logging.info('iter: {}\treward: {}\tq_loss: {}\telapse: {:.02f}'.format(iteration+1,\
                                                                            reward_mean,\
                                                                            loss.item(),\
                                                                            time.time()-start))

    if (iteration+1)%1000 == 0:
        torch.save(policy.state_dict(),  './exp/p'+str(iteration+1)+'.model')
        torch.save(q_func.state_dict(),  './exp/q'+str(iteration+1)+'.model')
        torch.save(p_optim.state_dict(), './exp/p_optim.state')
        torch.save(q_optim.state_dict(), './exp/q_optim.state')
