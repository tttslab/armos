# -*- coding: utf-8 -*-
import utils
import torch
import torch.nn.functional as F
import numpy as np
import logging

class VTLEnv():
    def __init__(
        self, 
        IN_SIZE,
        BATCH_SIZE,
        AUDIO_SEGMENT,
        FRAME_RATE_HZ,
        NUM_PARAL,
        dataset='training'
    ):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.IN_SIZE = IN_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.AUDIO_SEGMENT = AUDIO_SEGMENT
        self.FRAME_RATE_HZ = FRAME_RATE_HZ
        self.NUM_PARAL = NUM_PARAL

        self.loader = utils.Batch_generator(dataset, BATCH_SIZE)
        
        self.reset()
        
    def step(self, action):
        noise_act, normal_act = action

        ### Assume the duration is multiple of 10ms.
        ### For example, if the time length of input feature is 101, this sound is from 1000ms to 1009ms,
        ### but I regard this as 1000ms and ignore the last feature.
        inputs, length, _ = next(self.loader)

        feats = np.zeros((self.BATCH_SIZE, self.AUDIO_SEGMENT, self.IN_SIZE))
        for i in range(self.BATCH_SIZE):
            s_pos    = np.random.randint(length[i]-self.AUDIO_SEGMENT+1)
            feats[i] = inputs[i, s_pos:s_pos+self.AUDIO_SEGMENT,:]
        inputs = np.asarray(feats, dtype=np.float32)
        inputs            = torch.from_numpy(inputs).to(self.device)
        length            = np.asarray(length, dtype=np.int32) - 1
        length            = np.where(length > self.AUDIO_SEGMENT, self.AUDIO_SEGMENT, length)
        dur               = 1.0 * length / self.FRAME_RATE_HZ ### dimension is sec.

        ### Store parameters
        tractParams   = utils.trans_param(noise_act)[:, :, :24]
        glottisParams = utils.trans_param(noise_act)[:, :, -6:]
        r_calc_t      = utils.trans_param(normal_act.detach())[:, :, :24]
        r_calc_g      = utils.trans_param(normal_act.detach())[:, :, -6:]

        ### Reward calculation
        reward = utils.calc_reward(inputs, tractParams, glottisParams, length, dur, self.NUM_PARAL)
        r_calc = utils.calc_reward(inputs, r_calc_t,    r_calc_g,      length, dur, self.NUM_PARAL)
        self.reward_mean = 0
        for i in range(self.BATCH_SIZE):
            self.reward_mean += F.mse_loss(torch.from_numpy(r_calc[i][:length[i]]).to(self.device), inputs[i, :length[i]], reduction='none').mean(dim=1).sum()
        self.reward_mean /= length.sum()
        
        state = (inputs, length)
        return (state, reward, False, None)

    def render(self, iter, q_loss, sigma_init, time, mode="log"):
        logging.info('iter: {}\treward: {}\tq_loss: {}\tsigma:{}\telapse: {:.02f}'.format(iter+1,\
                                                                                      self.reward_mean,\
                                                                                      q_loss,\
                                                                                      sigma_init,\
                                                                                      time))

    def reset(self):
        self.reward_mean = 0
        inputs, length, _ = next(self.loader)

        # TODO: ここで何をやっているのかを正確に把握して関数化する
        feats = np.zeros((self.BATCH_SIZE, self.AUDIO_SEGMENT, self.IN_SIZE))
        for i in range(self.BATCH_SIZE):
            s_pos    = np.random.randint(length[i]-self.AUDIO_SEGMENT+1)
            feats[i] = inputs[i, s_pos:s_pos+self.AUDIO_SEGMENT,:]
        inputs = np.asarray(feats, dtype=np.float32)
        inputs            = torch.from_numpy(inputs).to(self.device)
        length            = np.asarray(length, dtype=np.int32) - 1
        length            = np.where(length > self.AUDIO_SEGMENT, self.AUDIO_SEGMENT, length)
        dur               = 1.0 * length / self.FRAME_RATE_HZ ### dimension is sec.

        return (inputs, length)