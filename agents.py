# -*- coding: utf-8 -*-
"""
@Time ： 2020/7/17 16:16
@Auth ： Kunfeng Li
@File ：agents.py
@IDE ：PyCharm

"""
import torch
import numpy as np
from policy.q_decom import Q_Decom


class Agents:
    def __init__(self, args, agent_id=None):
        self.args = args
        self.agent_id = agent_id
        self.q_decom_policy = ['qmix', 'vdn']
        if args.alg in self.q_decom_policy:
            self.policy = Q_Decom(args)
        else:
            raise Exception("算法不存在")

    def choose_action(self, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False):
        # 可供选择的动作
        avail_actions = np.nonzero(avail_actions_mask)[0]
        # agent索引转为独热编码
        onehot_agent_idx = np.zeros(self.args.n_agents)
        onehot_agent_idx[agent_idx] = 1.
        if self.args.last_action:
            # 在水平方向上平铺
            obs = np.hstack((obs, last_action))
        if self.args.reuse_network:
            obs = np.hstack((obs, onehot_agent_idx))
        hidden_state = self.policy.eval_hidden[:, agent_idx, :]
        # 转置
        obs = torch.Tensor(obs).unsqueeze(0)
        avail_actions_mask = torch.Tensor(avail_actions_mask).unsqueeze(0)
        # 是否使用 GPU
        if self.args.cuda:
            obs = obs.cuda()
            hidden_state = hidden_state.cuda()
        # 获取 Q(s, a)
        qsa, self.policy.eval_hidden[:, agent_idx, :] = self.policy.eval_rnn(obs, hidden_state)
        # 不可选的动作 q 值设为无穷小
        qsa[avail_actions_mask == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            return np.random.choice(avail_actions)
        return torch.argmax(qsa)

    def get_max_episode_len(self, batch):
        max_len = 0
        for episode in batch['padded']:
            length = episode.shape[0] - int(episode.sum())
            if length > max_len:
                max_len = length
        return int(max_len)

    def train(self, batch, train_step, epsilon=None):
        max_episode_len = self.get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_model_period == 0:
            self.policy.save_model(train_step)
