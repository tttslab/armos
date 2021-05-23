import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__()
    self.sigma_init   = sigma_init
    self.in_features  = in_features
    self.out_features = out_features

    weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-math.sqrt(3.0 / in_features), math.sqrt(3.0 / in_features)))
    bias   = nn.Parameter(torch.empty(out_features).uniform_(-math.sqrt(3.0 / in_features), math.sqrt(3.0 / in_features)))
    self.register_parameter('weight', weight)
    self.register_parameter('bias', bias)

  def forward(self, inputs, noise):
    device = inputs.device
    if noise:
      epsilon_weight = torch.randn(self.out_features, self.in_features).to(device) * self.sigma_init
      epsilon_bias   = torch.randn(self.out_features).to(device) * self.sigma_init
    else:
      epsilon_weight = torch.zeros(self.out_features, self.in_features).to(device)
      epsilon_bias = torch.zeros(self.out_features).to(device)
    return F.linear(inputs, self.weight + epsilon_weight, self.bias + epsilon_bias)

class stacked_BLSTM(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers, delta):
        super(stacked_BLSTM, self).__init__()
        self.blstm   = nn.LSTM(in_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc      = NoisyLinear(hidden_size*2, out_size, sigma_init=delta)
        self.sigmoid = nn.Sigmoid()
        self.relu    = nn.ReLU()

    def forward(self, inputs, length, noise):
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, length, batch_first=True)
        blstm_out, (h, c) = self.blstm(inputs)
        blstm_out, _      = nn.utils.rnn.pad_packed_sequence(blstm_out, batch_first=True)
        return self.sigmoid(self.fc(blstm_out, noise))

class Qfunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Qfunction, self).__init__()
        self.stat_embed = nn.Linear(state_dim, hidden_size)
        self.act_embed  = nn.Linear(action_dim, hidden_size)
        self.blstm      = nn.LSTM(hidden_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.out_q      = nn.Linear(hidden_size*2, state_dim)
        self.relu       = nn.ReLU()

    def forward(self, action, length):
        #s_emb           = self.stat_embed(state)
        a_emb           = self.act_embed(action)
        embeded         = nn.utils.rnn.pack_padded_sequence(a_emb, length, batch_first=True)
        q_value, (h, c) = self.blstm(embeded)
        q_value, _      = nn.utils.rnn.pad_packed_sequence(q_value, batch_first=True)
        #q_value         = torch.cat((h[-2], h[-1]), dim=1)
        q_value         = self.out_q(q_value)
        return q_value
