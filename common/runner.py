# -*- coding: utf-8 -*-
"""
@Time ： 2020/08/07 10:52
@Auth ： Kunfeng Li
@File ：runner.py
@IDE ：PyCharm

"""
import os
import numpy as np
from common.replay_buffer import ReplayBuffer
from agents import Agents
import time
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args, itr):
        # 获取参数
        # self.args = get_common_args()
        self.args = args

        # 获取环境
        self.env = env
        # 进程编号
        self.pid = itr

        self.agents = Agents(args, itr=itr)
        # 不复用网络，就会有多个agent，训练的时候共享参数，就是一个网络
        # if not self.args.reuse_network:
        #     self.agents = []
        #     for i in range(self.args.n_agents):
        #         self.agents.append(Agents(self.args, i))

        # self.rollout = RollOut(self.agents, self.args)

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

        # 保存结果和模型的位置，增加计数，帮助一次运行多个实例
        self.save_path = self.args.result_dir + '/' + self.args.alg + '/' + self.args.map + '/' + str(itr)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        print('runner 初始化')

    def generate_episode(self, episode_num, evaluate=False):
        # 为保存评价的回放做准备
        if self.args.replay_dir != '' and evaluate and episode_num == 0:
            self.env.close()
        # 变量初始化
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
            episode_buffer = {'o':            np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                              's':            np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'a':            np.zeros([self.args.episode_limit, self.args.n_agents, 1]),
                              'onehot_a':     np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'avail_a':      np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'r':            np.zeros([self.args.episode_limit, 1]),
                              'next_o':       np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                              'next_s':       np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'next_avail_a': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'done':         np.ones([self.args.episode_limit, 1]),
                              'padded':       np.ones([self.args.episode_limit, 1])
                              }
        # 开始进行一波 episode
        obs = self.env.get_obs()
        state = self.env.get_state()
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
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
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
                # 获取改变后的信息
                if not done:
                    next_obs = self.env.get_obs()
                    next_state = self.env.get_state()
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
        return episode_buffer, episode_reward, win

    def run(self):
        train_steps = 0
        early_stop = 10
        num_eval = 0
        self.max_win_rate = 0

        for itr in range(self.args.n_itr):
            # 收集 n_episodes 的数据
            episode_batch, _, _ = self.generate_episode(0)
            for key in episode_batch.keys():
                episode_batch[key] = np.array([episode_batch[key]])
            for e in range(1, self.args.n_episodes):
                episode, _, _ = self.generate_episode(e)
                for key in episode_batch.keys():
                    episode[key] = np.array([episode[key]])
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 添加到 replay buffer
            self.replay_buffer.store(episode_batch)
            # 训练
            if self.replay_buffer.size < self.args.batch_size * 12.5:
                # print('replay buffer 还没 batch size 大')
                continue
            for _ in range(self.args.train_steps):
                batch = self.replay_buffer.sample(self.args.batch_size)
                self.agents.train(batch, train_steps)
                # if self.args.reuse_network:
                #     self.agents.train(batch, train_steps)
                # else:
                #     for i in range(self.args.n_agents):
                #         self.agents[i].train(batch, train_steps)
                train_steps += 1
            # 周期性评价
            if itr % self.args.evaluation_period == 0:
                num_eval += 1
                print(f'进程 {self.pid}: {itr} / {self.args.n_itr}')
                win_rate, episodes_reward = self.evaluate()
                # 保存测试结果
                self.evaluate_itr.append(itr)
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
            _, episode_reward, win = self.generate_episode(itr, evaluate=True)
            episodes_reward.append(episode_reward)
            if win:
                win_number += 1
        return win_number/self.args.evaluate_num, episodes_reward

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
        np.save(self.save_path + '/win_rates.npy', self.win_rates)
        np.save(self.save_path + '/episodes_rewards.npy', self.episodes_rewards)

    def plot(self):
        """
        定期绘图
        :return:
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        win_x = np.array(self.evaluate_itr)[:, None]
        win_y = np.array(self.win_rates)[:, None]
        plot_win = pd.DataFrame(np.concatenate((win_x, win_y), axis=1), columns=['evaluate_itr', 'win_rates'])
        sns.lineplot(x="evaluate_itr", y="win_rates", data=plot_win, ax=ax1)

        ax2 = fig.add_subplot(212)
        reward_x = np.repeat(self.evaluate_itr, self.args.evaluate_num)[:, None]
        reward_y = np.array(self.episodes_rewards).flatten()[:, None]
        plot_reward = pd.DataFrame(np.concatenate((reward_x, reward_y), axis=1),
                                   columns=['evaluate_itr', 'episodes_rewards'])
        sns.lineplot(x="evaluate_itr", y="episodes_rewards", data=plot_reward, ax=ax2,
                     ci=68, estimator=np.median)

        # 格式化成2016-03-20-11_45_39形式
        tag = self.args.alg + '-' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # 如果已经有图片就删掉
        for filename in os.listdir(self.save_path):
            if filename.endswith('.png'):
                os.remove(self.save_path + '/' + filename)
        fig.savefig(self.save_path + "/%s.png" % tag)
        plt.close()

        # plt.figure()
        # plt.cla()
        # # 平均胜率图
        # plt.subplot(2, 1, 1)
        #
        # plt.plot(self.evaluate_itr, self.win_rates, c='blue')
        #
        # plt.xlabel('itr')
        # plt.ylabel('win_rates')
        # # 累加奖赏误差阴影曲线图
        # plt.subplot(2, 1, 2)
        # avg_rewards = np.array(self.episodes_rewards).mean(axis=1)
        #
        # plt.plot(self.evaluate_itr, avg_rewards, c='blue')
        #
        # up_error = []
        # down_error = []
        # for i in range(len(self.evaluate_itr)):
        #     up_bound = np.max(self.episodes_rewards[i])
        #     down_bound = np.min(self.episodes_rewards[i])
        #     up_error.append(up_bound - avg_rewards[i])
        #     down_error.append(avg_rewards[i] - down_bound)
        # # avg_rewards = np.array(avg_rewards)
        # up_error = np.array(up_error)
        # down_error = np.array(down_error)
        #
        # plt.fill_between(self.evaluate_itr, avg_rewards - down_error, avg_rewards + up_error,
        #                  color='blue', alpha=0.2)
        # plt.xlabel('itr')
        # plt.ylabel('episode_reward')
