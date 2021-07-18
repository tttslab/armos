# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import configparser
import librosa
from multiprocessing import Process
from multiprocessing import Queue
import TractSeqToWave_Func

conf = configparser.SafeConfigParser()
# TODO: 同じファイルを複数回読み込んでる問題
conf.read("./config/dpg_pn/config.ini")
IN_SIZE       = int(conf.get('main', 'in_size'))
command_list  = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', \
                 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', \
                 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
FEAT_DIR      = conf.get('main', 'feat_dir')
SAMPLING_RATE = int(conf.get('main', 'sampling_rate'))
frameRate_Hz  = int(conf.get('main', 'frameRate_Hz'))
FEAT_TYPE     = conf.get('main', 'feat_type')

### Convert 0-1 values to articulatory space
def trans_param(param):
    device = 'cuda' if param.is_cuda else 'cpu'
    min_val = [0.0, -3.228, -0.372, -7.0, -1.0, -1.102, 0.0, -0.1, 0.0, -3.194, -1.574, 0.873, \
               -1.574, -3.194, -1.574, -4.259, -3.228, -1.081, -1.081, -1.081, -1.081, 0.0, 0.0, 0.0, \
               40.0, 0.0, -0.0005, -0.0005, -0.00001, -40.0]
    max_val = [1.0, -1.85, 0.0, 0.0, 1.0, 2.205, 1.0, 1.0, 1.0, 2.327, 0.63, 3.2, \
               1.457, 2.327, 2.835, 1.163, 0.079, 1.081, 1.081, 1.081, 1.081, 0.3, 0.3, 0.3, \
               600.0, 2000.0, 0.003, 0.003, 0.00005, 0.0]
    min_val = torch.from_numpy(np.asarray(min_val, dtype=np.float32)).to(device)
    max_val = torch.from_numpy(np.asarray(max_val, dtype=np.float32)).to(device)
    return param * (max_val - min_val) + min_val

### This function is for multiprocessing.
### Assume attaching 5 pseudo frames to first and last of param sequence.
### This function returns feats
def synthesis(tractParams, glottisParams, duration, queue):
    synthesized = TractSeqToWave_Func.vtlSynthesize(tractParams, glottisParams, duration+frameRate_Hz/1000.0)
    synthesized = librosa.core.resample(synthesized.squeeze(), 22050, 16000)
    if FEAT_TYPE == 'mfcc':
        rec_feats   = librosa.feature.mfcc(synthesized[int(frameRate_Hz/2000.0*16000)-1:int((duration+frameRate_Hz/2000.0)*16000)],\
                                           sr=SAMPLING_RATE,\
                                           n_mfcc=IN_SIZE+1,\
                                           n_fft=int(SAMPLING_RATE/40.0),\
                                           hop_length=int(1.0 * SAMPLING_RATE/frameRate_Hz))
        rec_feats = np.asarray(rec_feats, dtype=np.float32)
        rec_feats = rec_feats[1:, :].T
        queue.put(rec_feats[:-1])
    elif FEAT_TYPE == 'fbank':
        rec_feats   = librosa.feature.melspectrogram(synthesized[int(frameRate_Hz/2000.0*16000)-1:int((duration+frameRate_Hz/2000.0)*16000)],\
                                                     sr=SAMPLING_RATE,\
                                                     n_mels=IN_SIZE,\
                                                     n_fft=int(SAMPLING_RATE/40.0),\
                                                     hop_length=int(1.0*SAMPLING_RATE/frameRate_Hz))
        rec_feats   = np.asarray(rec_feats, dtype=np.float32).T
        queue.put(rec_feats[:-1])
    elif FEAT_TYPE == 'logfbank':
        rec_feats   = librosa.feature.melspectrogram(synthesized[int(frameRate_Hz/2000.0*16000)-1:int((duration+frameRate_Hz/2000.0)*16000)],\
                                                     sr=SAMPLING_RATE,\
                                                     n_mels=IN_SIZE,\
                                                     n_fft=int(SAMPLING_RATE/40.0),\
                                                     hop_length=int(1.0*SAMPLING_RATE/frameRate_Hz))
        rec_feats   = np.log(np.asarray(rec_feats+1e-30, dtype=np.float32).T)
        queue.put(rec_feats[:-1])

### Reward definition
def reward(batch_rec_feats, inputs, length):
    device = 'cuda' if inputs.is_cuda else 'cpu'
    for i in range(inputs.shape[0]):
        if i == 0:
            r_tmp = -F.mse_loss(torch.from_numpy(batch_rec_feats[i]).to(device),\
                                inputs[i])
            reward = r_tmp.unsqueeze(dim=0)
        else:
            r_tmp = -F.mse_loss(torch.from_numpy(batch_rec_feats[i]).to(device),\
                                inputs[i])
            reward = torch.cat((reward, r_tmp.unsqueeze(dim=0)), dim=0)
    return reward

### Output is a list of each utterance's reward sequence.
def calc_reward(inputs, tractParams, glottisParams, length, dur, num_paral):
    batch_rec_feats = []
    t_param        = tractParams.to('cpu')
    # inputs.shape[0]は多分バッチサイズ
    # 最初と最後に擬似フレームを付与する
    t_tmp          = torch.zeros(inputs.shape[0], 5, 24)
    t_tmp[:, 2, :] = t_param[:, 0, :] / 3
    t_tmp[:, 3, :] = t_param[:, 0, :] * 2 / 3
    t_tmp[:, 4, :] = t_param[:, 0, :]
    t_param     = torch.cat((t_tmp, t_param), dim=1)
    for i in range(5):
        t_param     = torch.cat((t_param, t_param[:, -1, :].unsqueeze(dim=1)), dim=1)
    g_param     = glottisParams.to('cpu')
    g_tmp       = torch.zeros(inputs.shape[0], 5, 6)
    g_tmp[:, 2, :] = g_param[:, 0, :] / 3
    g_tmp[:, 3, :] = g_param[:, 0, :] * 2 / 3
    g_tmp[:, 4, :] = g_param[:, 0, :]
    g_param     = torch.cat((g_tmp, g_param), dim=1)
    for i in range(5):
        g_param     = torch.cat((g_param, g_param[:, -1, :].unsqueeze(dim=1)), dim=1)

    for num in range(int(inputs.shape[0] / num_paral)):
        rec_feats = []
        process   = []
        queue     = []
        for paral in range(num_paral):
            queue.append(Queue())
            process.append(Process(target=synthesis, args=(t_param[num*num_paral+paral, :(length[num*num_paral+paral]+10)].flatten().tolist(),\
                                                           g_param[num*num_paral+paral, :(length[num*num_paral+paral]+10)].flatten().tolist(),\
                                                           dur[num*num_paral+paral], queue[-1])))
        for proc in process:
            proc.start()
        for q in queue:
            # putが発生されるまでは処理が止まる
            rec_feats.append(q.get())
        for proc in process:
            proc.join()

        batch_rec_feats += rec_feats
    ### 計算された報酬ではなく生成された特徴量を返してる
    return batch_rec_feats
    #return reward(batch_rec_feats, inputs, length)

### Sort utterance in mini-batch.
def insert_index_descending_order(query, num_list):
    matching_list = list(filter(lambda x: x < query, num_list))
    if len(matching_list) == 0:
        return len(num_list)
    else:
        # もともと大きい順にソートされているのでmatching_listの先頭が新たな挿入位置
        return num_list.index(matching_list[0])

### Make mini-batch.
def Batch_generator(dataset, batch_size):
    # TODO: パスの修正 というかそもそもこの関数をここに置くのか問題
    if dataset == 'training':
        datalist_txt = open('config/dpg_pn/training_list.txt', 'r')
    elif dataset == 'testing':
        datalist_txt = open('datalist/testing_list.txt', 'r')
    elif dataset == 'validation':
        datalist_txt = open('datalist/validation_list.txt', 'r')

    datalist      = datalist_txt.read().strip().split('\n')
    shuffled_data = random.sample(datalist, len(datalist))
    datalist_txt.close()
    epoch         = 1

    while True:
        data_batch      = np.array([], dtype=np.float32)
        length_batch    = []
        duration_batch  = []
        numFrames_batch = []
        # バッチの中で最も長いフレーム数　なぜ大文字なのかは不明
        MAX_FRAME_LENGTH = 0
        for i in range(batch_size):
            sample  = shuffled_data.pop()
            feat    = np.load(FEAT_DIR + sample, allow_pickle=True)
            MAX_FRAME_LENGTH = len(feat) if MAX_FRAME_LENGTH < len(feat) else MAX_FRAME_LENGTH
            index   = insert_index_descending_order(len(feat), length_batch)
            if i == 0:
                data_batch = np.asarray([feat])
                data_batch = np.pad(data_batch, ((0, 0), (0, MAX_FRAME_LENGTH - data_batch.shape[1]), (0, 0)), mode='constant', constant_values=0)
            else:
                data_batch = np.pad(data_batch, ((0, 0), (0, MAX_FRAME_LENGTH - data_batch.shape[1]), (0, 0)), mode='constant', constant_values=0)
                data_batch = np.insert(data_batch, index, np.pad(feat, ((0, MAX_FRAME_LENGTH - len(feat)), (0, 0)), mode='constant', constant_values=0), axis=0)
            length_batch.insert(index, len(feat))
        data_batch  = np.asarray(data_batch,  dtype=np.float32)

        if len(shuffled_data) < batch_size:
            shuffled_data = random.sample(datalist, len(datalist)) + shuffled_data
            epoch        += 1

        # data_batch内には特徴量（バッチ内で最もフレーム数の大きなものに合わせて0詰めされており、降順に並んでいる。保存時に転置をとっていることにも注意）
        # length_batchはバッチ内の各データの0詰めされていない状態での長さを表している
        # epochはデータを何周したのか
        yield data_batch, length_batch, epoch

class OUNoise:
  def __init__(self, batch_size, size, mu=0., theta=0.01, sigma=0.01):
    self.batch_size = batch_size
    self.size  = size
    self.mu    = mu
    self.theta = theta
    self.sigma = sigma

  def reset(self, sigma):
    self.state = np.random.randn(self.batch_size, self.size)*sigma + self.mu

  def sample(self):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma*np.random.randn(self.batch_size, x.shape[1])
    self.state = x + dx
    return np.asarray(self.state, dtype=np.float32)
