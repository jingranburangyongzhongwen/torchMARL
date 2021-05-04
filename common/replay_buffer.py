# -*- coding: utf-8 -*-
"""
@Auth ： Kunfeng Li
@IDE ：PyCharm
"""
import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.current_size = 0
        # 记录多个episode的信息，线程安全，无需锁，但是内存占用和耗时较大
        # self.buffer = {'o': collections.deque(maxlen=self.args.buffer_size),
        #                's': collections.deque(maxlen=self.args.buffer_size),
        #                'a': collections.deque(maxlen=self.args.buffer_size),
        #                'onehot_a': collections.deque(maxlen=self.args.buffer_size),
        #                'avail_a': collections.deque(maxlen=self.args.buffer_size),
        #                'r': collections.deque(maxlen=self.args.buffer_size),
        #                'next_o': collections.deque(maxlen=self.args.buffer_size),
        #                'next_s': collections.deque(maxlen=self.args.buffer_size),
        #                'next_avail_a': collections.deque(maxlen=self.args.buffer_size),
        #                'done': collections.deque(maxlen=self.args.buffer_size),
        #                'padded': collections.deque(maxlen=self.args.buffer_size)
        #                }
        self.buffer = {
            'o': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
            's': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.state_shape]),
            'a': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, 1]),
            'onehot_a': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
            'avail_a': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
            'r': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, 1]),
            'next_o': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
            'next_s': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.state_shape]),
            'next_avail_a': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
            'done': np.ones(
                [self.args.buffer_size, self.args.episode_limit, 1]),
            'padded': np.ones(
                [self.args.buffer_size, self.args.episode_limit, 1])
        }
        self.current_idx = 0
        self.size = 0
        self.lock = threading.Lock()

    def sample(self, batch_size):
        """
        采样部分episode
        :param batch_size:
        :return:
        """
        temp_buffer = {}
        idxes = np.random.randint(0, self.size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idxes]
        return temp_buffer

    def store(self, episode_batch):
        with self.lock:
            # 这批数据有多少
            num = episode_batch['o'].shape[0]
            # 获取可以写入的位置
            idxes = self.get_idxes(num)
            # 存储经验
            for key in self.buffer.keys():
                self.buffer[key][idxes] = episode_batch[key]
            # 更新当前 buffer 的大小
            self.size = min(self.args.buffer_size, self.size + num)

    def get_idxes(self, num):
        """
        得到可以填充的索引数组
        :return: 索引数组
        """
        # 如果保存后不超过 buffer 的大小，则返回当前已填充到的索引+1开始 inc 长度的索引数组
        if self.current_idx + num <= self.args.buffer_size:
            idxes = np.arange(self.current_idx, self.current_idx + num)
            self.current_idx += num
        # 如果剩下位置不足以保存，但 current_idx 还没到末尾，则填充至末尾后，从头开始覆盖
        elif self.current_idx < self.args.buffer_size:
            overflow = num - (self.args.buffer_size - self.current_idx)
            idxes = np.concatenate([np.arange(self.current_idx, self.args.buffer_size),
                                    np.arange(0, overflow)])
            self.current_idx = overflow
        # 否则直接从头开始覆盖
        else:
            idxes = np.arange(0, num)
            self.current_idx = num
        return idxes

