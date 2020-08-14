# -*- coding: utf-8 -*-
"""
@Time ： 2020/07/25 16:52
@Auth ： Kunfeng Li
@File ：plot_compare.py
@IDE ：PyCharm

"""
from common.arguments import get_common_args
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    """
    从指定位置加载数据，方便后续作图
    :param path:
    :return:
    """
    return np.load(path + '/evaluate_itr.npy'), np.load(path + '/win_rates.npy'), np.load(path + '/episodes_rewards.npy')


if __name__ == '__main__':
    args = get_common_args()
    map = '3m'
    algs = ['qmix', 'vdn']
    colors = ['blue', 'k', 'r', 'tan', 'yellowgreen', 'darksage', 'g', 'c', 'purple']

    # plt.figure()
    # plt.cla()
    # # 平均胜率图
    # plt.subplot(2, 1, 1)
    # for i in range(len(algs)):
    #     evaluate_itr, win_rates, _ = load_results(args.result_dir + '/' + algs[i] + '/' + map)
    #     plt.plot(evaluate_itr, win_rates, c=colors[i], label=algs[i])

    # plt.xlabel('itr')
    # plt.ylabel('win_rates')
    # # 累加奖赏误差阴影曲线图
    # plt.subplot(2, 1, 2)
    # for i in range(len(algs)):
    #     evaluate_itr, _, episodes_rewards = load_results(args.result_dir + '/' + algs[i] + '/' + map)
    #     avg_rewards = np.array(episodes_rewards).mean(axis=1)
    #     plt.plot(evaluate_itr, avg_rewards, c=colors[i], label=algs[i])

    #     up_error = []
    #     down_error = []
    #     for j in range(len(evaluate_itr)):
    #         up_bound = np.max(episodes_rewards[j])
    #         down_bound = np.min(episodes_rewards[j])
    #         up_error.append(up_bound - avg_rewards[j])
    #         down_error.append(avg_rewards[j] - down_bound)
    #     # avg_rewards = np.array(avg_rewards)
    #     up_error = np.array(up_error)
    #     down_error = np.array(down_error)

    #     plt.fill_between(evaluate_itr, avg_rewards - down_error, avg_rewards + up_error,
    #                      color=colors[i], alpha=0.2)
    # plt.xlabel('itr')
    # plt.ylabel('episode_reward')

    # plt.legend()

    # 格式化成2016-03-20_11-45-39形式
    tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # 如果已经有图片就删掉
    for filename in os.listdir(args.result_dir):
        if filename.endswith('.png'):
            os.remove(args.result_dir + '/' + filename)
    plt.savefig(args.result_dir + "/%s.png" % tag)
    # plt.close()
