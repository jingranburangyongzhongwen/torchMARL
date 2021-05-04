# -*- coding: utf-8 -*-
"""
@Auth ： Kunfeng Li
@IDE ：PyCharm
"""
import os
import random

import numpy as np
from common.replay_buffer import ReplayBuffer
from agents import Agents
import time
import torch
from datetime import datetime, timedelta, timezone
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args, itr, seed):
        # 随机设置种子
        if seed is not None:
            self.setup_seed(seed)
        self.args = args

        # 获取环境
        self.env = env
        # 进程编号
        self.pid = itr

        self.replay_buffer = ReplayBuffer(self.args)

        self.win_rates = []
        '''
        这里，episode_reward 代表一个episode的累加奖赏，
        episodes_reward代表多个episode的累加奖赏，
        episodes_rewards代表多次评价的多个episode的累加奖赏
        '''
        self.episodes_rewards = []
        self.evaluate_itr = []

        self.max_win_rate = 0
        self.time_steps = 0

        # 保存结果和模型的位置，增加计数，帮助一次运行多个实例
        alg_dir = self.args.alg + '_' + str(self.args.epsilon_anneal_steps // 10000) + 'w' + '_' + \
                  str(self.args.target_update_period)
        self.alg_tag = '_' + self.args.optim

        if self.args.her:
            self.alg_tag += str(self.args.her)
            alg_dir += '_her=' + str(self.args.her)

        # self.save_path = self.args.result_dir + '/' + alg_dir + '/' + self.args.map + '/' + itr
        self.save_path = self.args.result_dir + '/' + self.args.map + '/' + alg_dir + '/' + itr
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.args.model_dir = args.model_dir + '/' + args.map + '/' + alg_dir + '/' + itr

        self.agents = Agents(args, itr=itr)
        print('step runner 初始化')
        if self.args.her:
            print('使用HER')

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def generate_episode(self, episode_num, evaluate=False):
        # 为保存评价的回放做准备
        if self.args.replay_dir != '' and evaluate and episode_num == 0:
            self.env.close()
        # 变量初始化，使用her需要记录goal
        self.env.reset()
        done = False
        info = None
        win = False

        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # epsilon 递减
        epsilon = 0 if evaluate else self.args.epsilon
        # epsilon 递减的方式
        if self.args.epsilon_anneal_scale == 'episode' or \
                (self.args.epsilon_anneal_scale == 'itr' and episode_num == 0):
            epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon

        # 记录一个episode的信息
        episode_buffer = None
        if not evaluate:
            episode_buffer = {'o': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                              's': np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'a': np.zeros([self.args.episode_limit, self.args.n_agents, 1]),
                              'onehot_a': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'avail_a': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'r': np.zeros([self.args.episode_limit, 1]),
                              'next_o': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                              'next_s': np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'next_avail_a': np.zeros(
                                  [self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'done': np.ones([self.args.episode_limit, 1]),
                              'padded': np.ones([self.args.episode_limit, 1])
                              }
        # 开始进行一波 episode
        states, former_states = [], []
        obs = self.env.get_obs()
        if self.args.her:
            obs = np.concatenate((obs, self.env.goal), axis=1)
        state = self.env.get_state()
        if self.args.her:
            states.append(self.env.state)
            former_states.append(self.env.former_states)
        avail_actions = []
        self.agents.policy.init_hidden(1)
        for agent_id in range(self.args.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)

        episode_reward = 0
        for step in range(self.args.episode_limit):
            if done:
                break
            else:
                actions, onehot_actions = [], []
                for agent_id in range(self.args.n_agents):
                    # avail_action = self.env.get_avail_agent_actions(agent_id)
                    action, _ = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                          avail_actions[agent_id], epsilon, evaluate)
                    # 得到该动作的独热编码
                    onehot_action = np.zeros(self.args.n_actions)
                    onehot_action[action] = 1
                    onehot_actions.append(onehot_action)
                    # 加入联合动作
                    actions.append(action)
                    # avail_actions.append(avail_action)
                    # 记录该动作
                    last_action[agent_id] = onehot_action
                # 对环境执行联合动作
                reward, done, info = self.env.step(actions)
                # 记录时间步
                if not evaluate:
                    self.time_steps += 1
                # 获取改变后的信息
                if not done:
                    next_obs = self.env.get_obs()
                    if self.args.her:
                        next_obs = np.concatenate((next_obs, self.env.goal), axis=1)
                    next_state = self.env.get_state()
                    if self.args.her:
                        states.append(self.env.state)
                        former_states.append(self.env.former_states)
                else:
                    next_obs = obs
                    next_state = state
                # 添加可得动作
                next_avail_actions = []
                for agent_id in range(self.args.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id)
                    next_avail_actions.append(avail_action)
                # 添加经验
                if not evaluate:
                    episode_buffer['o'][step] = obs
                    episode_buffer['s'][step] = state
                    episode_buffer['a'][step] = np.reshape(actions, [self.args.n_agents, 1])
                    episode_buffer['onehot_a'][step] = onehot_actions
                    episode_buffer['avail_a'][step] = avail_actions
                    episode_buffer['r'][step] = [reward]
                    episode_buffer['next_o'][step] = next_obs
                    episode_buffer['next_s'][step] = next_state
                    episode_buffer['next_avail_a'][step] = next_avail_actions
                    episode_buffer['done'][step] = [done]
                    episode_buffer['padded'][step] = [0.]

                # 更新变量
                episode_reward += reward
                obs = next_obs
                state = next_state
                avail_actions = next_avail_actions
                if self.args.epsilon_anneal_scale == 'step':
                    epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon

        # 是训练则记录新的epsilon
        if not evaluate:
            self.args.epsilon = epsilon
        # 获取对局信息
        if info.__contains__('battle_won'):
            win = True if done and info['battle_won'] else False
        if evaluate and episode_num == self.args.evaluate_num - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        if not evaluate and self.args.her:
            return episode_buffer, states, former_states
        return episode_buffer, episode_reward, win

    def run(self):
        train_steps = 0
        early_stop = 10
        num_eval = 0
        self.max_win_rate = 0
        self.time_steps = 0
        last_test_step = 0
        begin_time = None
        begin_step = None

        # for itr in range(self.args.n_itr):
        while self.time_steps < self.args.max_steps:
            if begin_step is None:
                begin_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8)))
                begin_step = self.time_steps
            # 收集 n_episodes 的数据
            if self.args.her:
                episode_batch, states, former_states = self.generate_episode(0)
                self.her_k(episode_batch, states, former_states)
            else:
                episode_batch, _, _ = self.generate_episode(0)
            for key in episode_batch.keys():
                episode_batch[key] = np.array([episode_batch[key]])
            for e in range(1, self.args.n_episodes):
                if self.args.her:
                    episode_batch, states, former_states = self.generate_episode(e)
                    self.her_k(episode_batch, states, former_states)
                else:
                    episode, _, _ = self.generate_episode(e)

                for key in episode_batch.keys():
                    episode[key] = np.array([episode[key]])
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 添加到 replay buffer
            self.replay_buffer.store(episode_batch)
            # 训练 TODO 12.5
            if self.replay_buffer.size < self.args.batch_size * self.args.bs_rate:
                print('replay buffer 还没 batch size * {} 大 ！'.format(self.args.bs_rate))
                begin_time = None
                begin_step = None
                continue
            for _ in range(self.args.train_steps):
                batch = self.replay_buffer.sample(self.args.batch_size)
                self.agents.train(batch, train_steps)
                train_steps += 1
            # 周期性评价
            # if itr % self.args.evaluation_period == 0:
            if (self.time_steps - last_test_step) / self.args.evaluation_steps_period >= 1.0:
                num_eval += 1
                last_test_step = self.time_steps
                print(f'进程 {self.pid}: {self.time_steps} step / {self.args.max_steps} steps')
                # print('幂为：{}'.format(self.agents.policy.power))
                win_rate, episodes_reward = self.evaluate()
                # 保存测试结果
                self.evaluate_itr.append(self.time_steps)
                self.win_rates.append(win_rate)
                self.episodes_rewards.append(episodes_reward)
                # 表现好的模型要额外保存
                if win_rate > self.max_win_rate:
                    self.max_win_rate = win_rate
                    self.agents.policy.save_model(str(win_rate))
                # 不时刻保存，从而减少时间花费
                if num_eval % 50 == 0:
                    self.save_results()
                    self.plot()
                    # 记录经历50次测试花费了多久
                    now = datetime.utcnow().astimezone(timezone(timedelta(hours=8)))
                    elapsed_time = now - begin_time
                    expected_remain_time = (elapsed_time / (self.time_steps - begin_step)) * \
                                           (self.args.max_steps - self.time_steps)
                    expected_end_time = now + expected_remain_time
                    print("预计还需: {}".format(str(expected_remain_time)))
                    print("预计结束时间为: {}".format(expected_end_time.strftime("%Y-%m-%d_%H-%M-%S")))
        # 最后把所有的都保存一下
        self.save_results()
        self.plot()
        self.env.close()

    def evaluate(self):
        """
        得到平均胜率和每次测试的累加奖赏，方便画误差阴影图
        :return:
        """
        win_number = 0
        episodes_reward = []
        for itr in range(self.args.evaluate_num):
            if self.args.didactic:
                episode_reward, win = self.get_eval_qtot()
            else:
                _, episode_reward, win = self.generate_episode(itr, evaluate=True)
            episodes_reward.append(episode_reward)
            if win:
                win_number += 1
        return win_number / self.args.evaluate_num, episodes_reward

    def save_results(self):
        """
        保存数据，方便后面多种算法结果画在一张图里比较
        :return:
        """
        # 如果已经有图片就删掉
        for filename in os.listdir(self.save_path):
            if filename.endswith('.npy'):
                os.remove(self.save_path + '/' + filename)
        np.save(self.save_path + '/evaluate_itr.npy', self.evaluate_itr)
        if self.args.didactic and self.args.power is None and 'strapped' in self.args.alg:
            np.save(self.save_path + '/train_steps.npy', self.agents.policy.train_steps)
            np.save(self.save_path + '/differences.npy', self.agents.policy.differences)
        else:
            np.save(self.save_path + '/win_rates.npy', self.win_rates)
        np.save(self.save_path + '/episodes_rewards.npy', self.episodes_rewards)

    def plot(self):
        """
        定期绘图
        :return:
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        if self.args.didactic and self.args.power is None and 'strapped' in self.args.alg:
            win_x = np.array(self.agents.policy.train_steps)[:, None] / 1000000.
            win_y = np.array(self.agents.policy.differences)[:, None]
            plot_win = pd.DataFrame(np.concatenate((win_x, win_y), axis=1), columns=['T (mil)', self.args.which_diff])
            sns.lineplot(x="T (mil)", y=self.args.which_diff, data=plot_win, ax=ax1)
        else:
            win_x = np.array(self.evaluate_itr)[:, None] / 1000000.
            win_y = np.array(self.win_rates)[:, None]
            plot_win = pd.DataFrame(np.concatenate((win_x, win_y), axis=1), columns=['T (mil)', 'Test Win'])
            sns.lineplot(x="T (mil)", y="Test Win", data=plot_win, ax=ax1)

        ax2 = fig.add_subplot(212)
        reward_x = np.repeat(self.evaluate_itr, self.args.evaluate_num)[:, None] / 1000000.
        reward_y = np.array(self.episodes_rewards).flatten()[:, None]
        plot_reward = pd.DataFrame(np.concatenate((reward_x, reward_y), axis=1),
                                   columns=['T (mil)', 'Median Test Returns'])
        sns.lineplot(x="T (mil)", y="Median Test Returns", data=plot_reward, ax=ax2,
                     ci='sd', estimator=np.median)
        plt.tight_layout()
        # 格式化成2016-03-20-11_45_39形式
        # tag = self.args.alg + '-' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        tag = self.args.alg + '_' + str(self.args.target_update_period)
        # if 'averaged' in self.args.alg:
        tag += (self.alg_tag + '_' +
                datetime.utcnow().astimezone(
                    timezone(timedelta(hours=8))
                ).strftime("%Y-%m-%d_%H-%M-%S"))
        # 如果已经有图片就删掉
        for filename in os.listdir(self.save_path):
            if filename.endswith('.png'):
                os.remove(self.save_path + '/' + filename)
        fig.savefig(self.save_path + "/%s.png" % tag)
        plt.close()

    def get_eval_qtot(self):
        """
        得到eval qtot
        """
        self.env.reset()

        all_last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        # 开始进行一波 episode
        all_obs = self.env.get_obs()
        state = self.env.get_state()
        avail_actions = []
        self.agents.policy.init_hidden(1)
        eval_qs = []
        actions = []
        one_hot_actions = []
        hidden_evals = None
        for agent_idx in range(self.args.n_agents):
            obs = all_obs[agent_idx]
            last_action = all_last_action[agent_idx]
            avail_action = self.env.get_avail_agent_actions(agent_idx)
            avail_actions.append(avail_action)

            onehot_agent_idx = np.zeros(self.args.n_agents)
            onehot_agent_idx[agent_idx] = 1.
            if self.args.last_action:
                # 在水平方向上平铺
                obs = np.hstack((obs, last_action))
            if self.args.reuse_network:
                obs = np.hstack((obs, onehot_agent_idx))
            hidden_state = self.agents.policy.eval_hidden[:, agent_idx, :]
            # 转置
            obs = torch.Tensor(obs).unsqueeze(0)
            # 是否使用 GPU
            if self.args.cuda:
                obs = obs.cuda()
                hidden_state = hidden_state.cuda()
            # 获取 Q(s, a)
            qsa, hidden_eval = self.agents.policy.eval_rnn(obs, hidden_state)
            qsa[avail_action == 0.0] = -float("inf")

            eval_qs.append(torch.max(qsa))

            action = torch.argmax(qsa)
            actions.append(action)

            onehot_action = np.zeros(self.args.n_actions)
            onehot_action[action] = 1
            one_hot_actions.append(onehot_action)
            if hidden_evals is None:
                hidden_evals = hidden_eval
            else:
                hidden_evals = torch.cat([hidden_evals, hidden_eval], dim=0)

        s = torch.Tensor(state)
        eval_qs = torch.Tensor(eval_qs).unsqueeze(0)
        actions = torch.Tensor(actions).unsqueeze(0)
        one_hot_actions = torch.Tensor(one_hot_actions).unsqueeze(0)
        hidden_evals = hidden_evals.unsqueeze(0)
        # 是否使用GPU
        if self.args.cuda:
            s = s.cuda()
            eval_qs = eval_qs.cuda()
            actions = actions.cuda()
            one_hot_actions = one_hot_actions.cuda()
            hidden_evals = hidden_evals.cuda()
        # 计算Q_tot
        eval_q_total = None
        if self.args.alg == 'qatten':
            eval_q_total, _, _ = self.agents.policy.eval_mix_net(eval_qs, s, actions)
        elif self.args.alg == 'qmix' \
                or 'wqmix' in self.args.alg \
                or 'strapped' in self.args.alg:
            eval_q_total = self.agents.policy.eval_mix_net(eval_qs, s)
        elif 'dmaq' in self.args.alg:
            if self.args.alg == "dmaq_qatten":
                ans_chosen, _, _ = self.agents.policy.mixer(eval_qs, s, is_v=True)
                ans_adv, _, _ = self.agents.policy.mixer(eval_qs, s, actions=one_hot_actions,
                                                         max_q_i=eval_qs, is_v=False)
                eval_q_total = ans_chosen + ans_adv
            else:
                ans_chosen = self.agents.policy.mixer(eval_qs, s, is_v=True)
                ans_adv = self.agents.policy.mixer(eval_qs, s, actions=one_hot_actions,
                                                   max_q_i=eval_qs, is_v=False)
                eval_q_total = ans_chosen + ans_adv
        elif self.args.alg == 'qtran_base':
            one_hot_actions = one_hot_actions.unsqueeze(0)
            hidden_evals = hidden_evals.unsqueeze(0)
            eval_q_total = self.agents.policy.eval_joint_q(s, hidden_evals, one_hot_actions)

        eval_q_total = eval_q_total.squeeze().item()
        return eval_q_total, 0

    def her_k(self, episode, states, former_states):
        import copy
        for _ in range(self.args.her):
            episode_buffer = {'o': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                              's': np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'a': np.zeros([self.args.episode_limit, self.args.n_agents, 1]),
                              'onehot_a': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'avail_a': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'r': np.zeros([self.args.episode_limit, 1]),
                              'next_o': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                              'next_s': np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'next_avail_a': np.zeros(
                                  [self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'done': np.ones([self.args.episode_limit, 1]),
                              'padded': np.ones([self.args.episode_limit, 1])
                              }
            # 重新生成goal，order等信息
            self.env.reset()
            # 使用新生成的goals重构整个episode
            for i in range(len(episode)):
                reward = self.env.get_reward(states[i], former_states[i])
                done = episode['done'][i]
                if reward >= 0:
                    reward = 0
                    done = True
                episode_buffer['o'][i] = episode['o'][i]
                episode_buffer['o'][i, :, -2:] = np.array(self.env.goal)[:]
                episode_buffer['s'][i] = episode['s'][i]
                episode_buffer['a'][i] = episode['a'][i]
                episode_buffer['onehot_a'][i] = episode['onehot_a'][i]
                episode_buffer['avail_a'][i] = episode['avail_a'][i]
                episode_buffer['r'][i] = [reward]
                episode_buffer['next_o'][i] = episode['next_o'][i]
                episode_buffer['next_o'][i, :, -2:] = np.array(self.env.goal)[:]
                episode_buffer['next_s'][i] = episode['next_s'][i]
                episode_buffer['next_avail_a'][i] = episode['next_avail_a'][i]
                episode_buffer['done'][i] = [done]
                episode_buffer['padded'][i] = [0.]
                if done:
                    break
            for key in episode_buffer.keys():
                episode_buffer[key] = np.array([episode_buffer[key]])
            self.replay_buffer.store(episode_buffer)
