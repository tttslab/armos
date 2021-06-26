# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import networks

class DPGAgent:
    def __init__(
        self, 
        IN_SIZE, 
        OUT_SIZE, 
        P_HIDDEN_SIZE, 
        P_NUM_LAYERS, 
        DELTA, 
        SIGMA_INIT,
        ADJUST_STEP,
        P_LEARNING_RATE,
        Q_HIDDEN_SIZE,
        Q_LEARNING_RATE,
        BATCH_SIZE
    ):
        self.iteration = 0
        self.DELTA = DELTA
        self.ADJUST_STEP = ADJUST_STEP
        self.BATCH_SIZE = BATCH_SIZE

        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy       = networks.stacked_BLSTM(IN_SIZE, OUT_SIZE, P_HIDDEN_SIZE, P_NUM_LAYERS, SIGMA_INIT).to(self.device)
        self.q_func       = networks.Qfunction(IN_SIZE, OUT_SIZE, Q_HIDDEN_SIZE).to(self.device)
        # self.ou_noise     = utils.OUNoise(BATCH_SIZE, OUT_SIZE) # 使われてない
        self.loss_fun     = nn.MSELoss(reduction='none')
        self.p_optim      = torch.optim.SGD(self.policy.parameters(), lr=P_LEARNING_RATE)
        self.q_optim      = torch.optim.Adam(self.q_func.parameters(), lr=Q_LEARNING_RATE)

        self.policy.load_state_dict(torch.load('exp/dpg_pn/pretrain.model'))
        #p_optim.load_state_dict(torch.load('exp/p_optim.state'))
        #q_func.load_state_dict(torch.load('exp/q1000.model'))
        #q_optim.load_state_dict(torch.load('exp/q_optim.state'))
        
        self.noise_act = None
        self.normal_act = None
        self.inputs = None
        self.length = None


    def act(self, state):
        self.policy.train()
        self.q_func.train()

        self.inputs, self.length = state
        self.noise_act  = self.policy(self.inputs, self.length.tolist(), True)
        self.normal_act = self.policy(self.inputs, self.length.tolist(), False)

        if (self.iteration+1) % self.ADJUST_STEP == 0:
            if F.mse_loss(self.noise_act, self.normal_act) < self.DELTA**2:
                self.policy.fc.sigma_init *= 1.01
            else:
                self.policy.fc.sigma_init /= 1.01
        
        if (self.iteration+1)%1000 == 0:
            self._save()

        self.iteration += 1
        
        return (self.noise_act, self.normal_act)


    def update(self, reward):
        q_value = self.q_func(self.noise_act, self.length.tolist())
        loss    = 0
        for i in range(self.BATCH_SIZE):
            loss += self.loss_fun(q_value[i,:self.length[i]], torch.from_numpy(reward[i][:self.length[i]]).to( self.device)).mean(dim=1).sum()
        loss /= self.length.sum()
        self.q_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_func.parameters(), 10.0)
        self.q_optim.step()

        ### update policy for critic
        reconst =  self.q_func(self.normal_act, self.length.tolist())
        p_loss  = 0
        for i in range(self.BATCH_SIZE):
            p_loss += self.loss_fun(reconst[i, :self.length[i]], self.inputs[i, :self.length[i]]).mean(dim=1).sum()
        p_loss /= self.length.sum()
        self.p_optim.zero_grad()
        p_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.p_optim.step()

        return (loss.item(), self.policy.fc.sigma_init)


    def _save(self):
        torch.save(self.policy.state_dict(),  './exp/dpg_pn/p'+str(self.iteration+1)+'.model')
        torch.save(self.q_func.state_dict(),  './exp/dpg_pn/q'+str(self.iteration+1)+'.model')
        torch.save(self.p_optim.state_dict(), './exp/dpg_pn/p_optim.state')
        torch.save(self.q_optim.state_dict(), './exp/dpg_pn/q_optim.state')
