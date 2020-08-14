# -*- coding: utf-8 -*-
"""
@Time ： 2020/08/13 21:52
@Auth ： Kunfeng Li
@File ：wqmix_q_star.py
@IDE ：PyCharm

"""
import torch.nn as nn
import torch.nn.functional as F
from network.base_net import RNN


class QStar(nn.Module):
    # 因为所有agent共享这个网络，因此input_shape = obs_shape + n_actions + n_agents
    def __init__(self, args):
        super(QStar, self).__init__()
        self.args = args
        input_shape = self.args.obs_shape
        # 调整RNN输入维度
        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.args.n_agents
        # 设置网络
        self.agent = RNN(input_shape, args)

        self.fc = nn.Sequential(nn.Linear(input_shape, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1))

    def forward(self, qsas, states):
        episode_num = qsas.size(0)
        qsas = qsas.view(-1, 1, self.args.n_agents)
        states = states.reshape(-1, self.args.state_shape)

        # w1 = torch.abs(self.hyper_w1(states))
        # w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)
        # b1 = self.hyper_b1(states).view(-1, 1, self.args.qmix_hidden_dim)
        #
        # hidden = F.elu(torch.bmm(qsas, w1) + b1)
        #
        # w2 = torch.abs(self.hyper_w2(states)).view(-1, self.args.qmix_hidden_dim, 1)
        # b2 = self.hyper_b2(states).view(-1, 1, 1)
        #
        # q_total = torch.bmm(hidden, w2) + b2
        q_star = self.fc(h)
        q_star = q_star.view(episode_num, -1, 1)
        return q_star
