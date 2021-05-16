import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
import configparser
import librosa
import sys

conf = configparser.SafeConfigParser()
conf.read("config.ini")
IN_SIZE             = int(conf.get('main', 'in_size'))
OUT_SIZE            = int(conf.get('main', 'out_size'))
P_HIDDEN_SIZE       = int(conf.get('actor', 'hidden_size'))
P_NUM_LAYERS        = int(conf.get('actor', 'num_layers'))
Q_HIDDEN_SIZE       = int(conf.get('critic', 'hidden_size'))
BATCH_SIZE          = 1
SAMPLING_RATE       = int(conf.get('main', 'sampling_rate'))
frameRate_Hz        = int(conf.get('main', 'frameRate_Hz'))
FEAT_TYPE           = conf.get('main', 'feat_type')

device       = 'cuda' if torch.cuda.is_available() else 'cpu'
policy       = models.stacked_BLSTM(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, P_NUM_LAYERS, 0).to(device)
q_func       = models.Qfunction(IN_SIZE, OUT_SIZE, Q_HIDDEN_SIZE).to(device)

num = sys.argv[1]
policy.load_state_dict(torch.load('exp/p' + num + '.model'))

tract_dict = np.load('testdata/testdata.npy')
target  = torch.from_numpy(tract_dict).float().to(device)
for i in range(101):
    audio, sr = librosa.load('testdata/' + str(i+1) + '.wav', sr=SAMPLING_RATE)

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
    feats = feats.T[:-1]
    if i == 0:
        inputs = torch.from_numpy(feats).unsqueeze(dim=0)
        length = [inputs.shape[1]]
    else:
        inputs = torch.cat((inputs, torch.from_numpy(feats).unsqueeze(dim=0)), dim=0)
        length.append(inputs.shape[1])
inputs = inputs.to(device)

rmse = 0
with torch.no_grad():
    policy.eval()
    q_func.eval()
    action = policy(inputs, length, False)[:, :, :24]

    loss = F.mse_loss(action, target, reduction='none')
    loss = loss.mean(dim=1)
    #print(torch.sqrt(loss.mean(dim=1)))
    #print(torch.sqrt(loss.mean()).item())
    act_dist = torch.sqrt(loss.mean(dim=0)).to('cpu').unsqueeze(dim=0).numpy()
    with open('log/eval.csv', 'a') as f:
        np.savetxt(f, act_dist, delimiter=',')
    
