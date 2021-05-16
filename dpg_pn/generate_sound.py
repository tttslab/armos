import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
import utils
import configparser
import TractSeqToWave_synthesis
import librosa
import sys

### Hyperparameters
conf = configparser.SafeConfigParser()
conf.read("config.ini")
IN_SIZE             = int(conf.get('main', 'in_size'))
OUT_SIZE            = int(conf.get('main', 'out_size'))
P_HIDDEN_SIZE       = int(conf.get('actor', 'hidden_size'))
P_NUM_LAYERS        = int(conf.get('actor', 'num_layers'))
Q_HIDDEN_SIZE       = int(conf.get('critic', 'hidden_size'))
SAMPLING_RATE       = int(conf.get('main', 'sampling_rate'))
frameRate_Hz        = int(conf.get('main', 'frameRate_Hz'))
FEAT_TYPE           = conf.get('main', 'feat_type')

### Condition Setting
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
policy       = models.stacked_BLSTM(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, P_NUM_LAYERS, 0).to(device)
policy.load_state_dict(torch.load(sys.argv[2]))

audio, sr = librosa.load(sys.argv[1], sr=SAMPLING_RATE)
if FEAT_TYPE == 'mfcc':
    feats = librosa.feature.mfcc(audio, sr=SAMPLING_RATE, n_mfcc=IN_SIZE+1, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(1.0*SAMPLING_RATE/frameRate_Hz))
    feats = np.asarray(feats, dtype=np.float32)
    feats = feats[1:, :]
elif FEAT_TYPE == 'fbank':
    feats = librosa.feature.melspectrogram(audio, sr=SAMPLING_RATE, n_mels=IN_SIZE, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(1.0*SAMPLING_RATE/frameRate_Hz))
    feats = np.asarray(feats, dtype=np.float32)
elif FEAT_TYPE == 'logfbank':
    feats = librosa.feature.melspectrogram(audio, sr=SAMPLING_RATE, n_mels=IN_SIZE, n_fft=int(SAMPLING_RATE/40.0), hop_length=int(1.0*SAMPLING_RATE/frameRate_Hz))
    feats = np.log(np.asarray(feats+1e-30, dtype=np.float32))
feats = torch.from_numpy(feats.T[:-1]).to(device)

with torch.no_grad():
    policy.eval()

    ### Assume the duration is multiple of 10ms.
    ### For example, if the time length of input feature is 101, this sound is from 1000ms to 1009ms,
    ### but I regard this as 1000ms and ignore the last feature.
    inputs = feats.unsqueeze(dim=0)
    length = [inputs.shape[1]]
    length = np.asarray(length, dtype=np.int32)
    dur    = 1.0*length / frameRate_Hz ### dimension is sec.

    ### Get action
    action  = policy(inputs, length.tolist(), False)
    action  = utils.trans_param(action)

    ### Store parameters
    tractParams   = action[0, :length[0], :24].reshape(1, length[0], 24)
    glottisParams = action[0, :length[0], -6:].reshape(1, length[0], 6)
    t_param        = tractParams.to('cpu')
    t_tmp          = torch.zeros(1, 5, 24)
    t_tmp[:, 2, :] = t_param[:, 0, :] / 3
    t_tmp[:, 3, :] = t_param[:, 0, :] * 2 / 3
    t_tmp[:, 4, :] = t_param[:, 0, :]
    t_param     = torch.cat((t_tmp, t_param), dim=1)
    for i in range(5):
        t_param = torch.cat((t_param, t_param[:, -1, :].unsqueeze(dim=1)), dim=1)
    g_param     = glottisParams.to('cpu')
    g_tmp       = torch.zeros(1, 5, 6)
    g_tmp[:, 2, :] = g_param[:, 0, :] / 3
    g_tmp[:, 3, :] = g_param[:, 0, :] * 2 / 3
    g_tmp[:, 4, :] = g_param[:, 0, :]
    g_param     = torch.cat((g_tmp, g_param), dim=1)
    for i in range(5):
        g_param = torch.cat((g_param, g_param[:, -1, :].unsqueeze(dim=1)), dim=1)

    TractSeqToWave_synthesis.vtlSynthesize(t_param[0].flatten().tolist(), g_param[0].flatten().tolist(), dur[0]+frameRate_Hz/1000.0)
