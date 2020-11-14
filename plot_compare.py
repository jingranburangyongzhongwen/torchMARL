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
import time
import seaborn as sns
import pandas as pd
import math


def load_results(alg, map, i):
    """
    从指定位置加载数据，方便后续作图
    :param path:
    :return:
    """
    path = '../Experiments/results/' + alg + '/' + map + '/' + str(i)
    return np.load(path + '/evaluate_itr.npy') / 1000., np.load(path + '/win_rates.npy'), np.load(path + '/episodes_rewards.npy')


def plot_one_run_compare(which, maps='3m', how_many=None, k=None, both=False, show=True):
    args = get_common_args()
    if k is None:
        algs = [which + '-k5', which + '-k10', which + '-k15']
    else:
        algs = ['Ensemble-k' + str(k), 'Averaged-k' + str(k)]
    # colors = ['blue', 'k', 'r', 'tan', 'yellowgreen', 'darksage', 'g', 'c', 'purple']

    evaluate_itr, win_rates, episodes_rewards = load_results('QMIX', maps, 1)
    alg = np.full(evaluate_itr.shape, fill_value='QMIX')
    win_compare = pd.DataFrame(np.concatenate((evaluate_itr[:, None], alg[:, None], win_rates[:, None]), axis=1),
                               columns=['Iterations (k)', 'Methods', 'Median win rates per episode'])

    evaluate_itr = np.repeat(evaluate_itr, args.evaluate_num)[:, None]
    episodes_rewards = episodes_rewards.flatten()[:, None]
    alg = np.full(evaluate_itr.shape, fill_value='QMIX')
    rewards_compare = pd.DataFrame(np.concatenate((evaluate_itr, alg, episodes_rewards), axis=1),
                                   columns=['Iterations (k)', 'Methods', 'Median reward per episode'])

    for i in range(len(algs)):
        evaluate_itr, win_rates, episodes_rewards = load_results(algs[i], maps, 1)
        if how_many is not None:
            internal = math.ceil(evaluate_itr.shape[0] / how_many)
            indxes = np.arange(0, evaluate_itr.shape[0], internal, dtype=np.int)
            evaluate_itr = np.array(evaluate_itr)[indxes:]
            win_rates = win_rates[indxes:]
            episodes_rewards = episodes_rewards[indxes:]
        alg = np.full(evaluate_itr.shape, fill_value=algs[i])
        tmp_win = pd.DataFrame(np.concatenate((evaluate_itr[:, None], alg[:, None], win_rates[:, None]), axis=1),
                               columns=['Iterations (k)', 'Methods', 'Median win rates per episode'])

        evaluate_itr = np.repeat(evaluate_itr, args.evaluate_num)[:, None]
        episodes_rewards = episodes_rewards.flatten()[:, None]
        alg = np.full(evaluate_itr.shape, fill_value=algs[i])
        tmp_reward = pd.DataFrame(np.concatenate((evaluate_itr, alg, episodes_rewards), axis=1),
                                  columns=['Iterations (k)', 'Methods', 'Median reward per episode'])
        win_compare = pd.concat([win_compare, tmp_win], axis=0)
        rewards_compare = pd.concat([rewards_compare, tmp_reward], axis=0)

    win_compare[['Iterations (k)', 'Median win rates per episode']] = win_compare[['Iterations (k)', 'Median win rates per episode']]\
        .apply(pd.to_numeric)
    rewards_compare[['Iterations (k)', 'Median reward per episode']] = rewards_compare[['Iterations (k)', 'Median reward per episode']]\
        .apply(pd.to_numeric)

    sns.set(context='paper', style='darkgrid')
    fig = plt.figure()
    plt.title(maps)
    if both:
        ax1 = fig.add_subplot(211)
        sns.lineplot(x="Iterations (k)", y="Median win rates per episode", hue='Methods', data=win_compare, ax=ax1,
                     ci='sd', estimator=np.median)

        ax2 = fig.add_subplot(212)
        sns.lineplot(x="Iterations (k)", y="Median reward per episode", hue='Methods', data=rewards_compare, ax=ax2,
                     ci=68, estimator=np.median)
    else:
        sns.lineplot(x="Iterations (k)", y="Median reward per episode", hue='Methods', data=rewards_compare,
                     ci=68, estimator=np.median)
    # 格式化成2016-03-20_11-45-39形式
    tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if show:
        plt.show()
    else:
        # palette=colors[:len(algs)+1] dpi=300
        if k is None:
            plt.savefig('../Experiments/reward_compare/' + which + '_' + maps + '_%s.png' % tag, dpi=350, bbox_inches='tight')
        else:
            plt.savefig('../Experiments/reward_compare/' + maps + '_%s.png' % tag, dpi=350, bbox_inches='tight')


def plot_multi_run_compare(which='ensemble', maps='3m', num=3, how_many=None, k=None, show=True):
    if k is None:
        algs = ['QMIX', which + '-k5', which + '-k10', which + '-k15']
    else:
        algs = ['QMIX', 'Ensemble-k' + str(k[0]), 'Averaged-k' + str(k[1])]
    # colors = ['blue', 'k', 'r', 'tan', 'yellowgreen', 'darksage', 'g', 'c', 'purple']
    win_compare = None
    for n in range(num):
        for i in range(len(algs)):
            evaluate_itr, win_rates, episodes_rewards = load_results(algs[i], maps, n)
            if how_many is not None:
                internal = math.ceil(evaluate_itr.shape[0] / how_many)
                indxes = np.arange(0, evaluate_itr.shape[0], internal, dtype=np.int)
                evaluate_itr = evaluate_itr[indxes]
                win_rates = win_rates[indxes]
            alg = np.full(evaluate_itr.shape, fill_value=algs[i])
            tmp_win = pd.DataFrame(np.concatenate((evaluate_itr[:, None], alg[:, None], win_rates[:, None]), axis=1),
                                   columns=['Iterations (k)', 'Methods', 'Median win rates per episode'])
            if win_compare is None:
                win_compare = tmp_win
            else:
                win_compare = pd.concat([win_compare, tmp_win], axis=0)

    win_compare[['Iterations (k)', 'Median win rates per episode']] = \
        win_compare[['Iterations (k)', 'Median win rates per episode']].apply(pd.to_numeric)

    sns.set(context='paper', style='darkgrid')
    fig = plt.figure()
    plt.title(maps)

    sns.lineplot(x="Iterations (k)", y="Median win rates per episode", hue='Methods', data=win_compare,
                 ci='sd', estimator=np.median)
    # 格式化成2016-03-20_11-45-39形式
    tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if show:
        plt.show()
    else:
        # palette=colors[:len(algs)+1] dpi=300
        if k is None:
            plt.savefig('../Experiments/reward_compare/' + which + '_' + maps + '_%s.png' % tag, dpi=350, bbox_inches='tight')
        else:
            plt.savefig('../Experiments/reward_compare/' + maps + '_%s.png' % tag, dpi=350, bbox_inches='tight')


def plot_reward_compare(which='averaged', maps='3m', n_run=None, how_many=None, k=None, show=True):
    if k is None:
        algs = ['QMIX', which + '-k5', which + '-k10', which + '-k15']
    else:
        algs = ['QMIX', 'Ensemble-k' + str(k[0]), 'Averaged-k' + str(k[1])]
    # colors = ['blue', 'k', 'r', 'tan', 'yellowgreen', 'darksage', 'g', 'c', 'purple']

    rewards_compare = None
    for i in range(len(algs)):
        j = 1
        if not n_run is None:
            j = n_run[i]
        evaluate_itr, win_rates, episodes_rewards = load_results(algs[i], maps, j)
        if how_many is not None:
            internal = math.ceil(evaluate_itr.shape[0] / how_many)
            indxes = np.arange(0, evaluate_itr.shape[0], internal, dtype=np.int)
            evaluate_itr = evaluate_itr[indxes]
            episodes_rewards = episodes_rewards[indxes]
        evaluate_itr = np.repeat(evaluate_itr, episodes_rewards.shape[1])[:, None]
        episodes_rewards = episodes_rewards.flatten()[:, None]
        alg = np.full(evaluate_itr.shape, fill_value=algs[i])
        tmp_reward = pd.DataFrame(np.concatenate((evaluate_itr, alg, episodes_rewards), axis=1),
                                  columns=['Iterations (k)', 'Methods', 'Median reward per episode'])
        if rewards_compare is None:
            rewards_compare = tmp_reward
        else:
            rewards_compare = pd.concat([rewards_compare, tmp_reward], axis=0)

    rewards_compare[['Iterations (k)', 'Median reward per episode']] = rewards_compare[['Iterations (k)', 'Median reward per episode']]\
        .apply(pd.to_numeric)

    sns.set(context='paper', style='darkgrid')
    fig = plt.figure()
    plt.title(maps)
    sns.lineplot(x="Iterations (k)", y="Median reward per episode", hue='Methods', data=rewards_compare,
                 ci=68, estimator=np.median)
    # 格式化成2016-03-20_11-45-39形式
    tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if show:
        plt.show()
    else:
        # palette=colors[:len(algs)+1] dpi=300
        if k is None:
            plt.savefig('../Experiments/reward_compare/' + which + '_' + maps + '_%s.png' % tag, dpi=350, bbox_inches='tight')
        else:
            plt.savefig('../Experiments/reward_compare/' + maps + '_%s.png' % tag, dpi=350, bbox_inches='tight')


if __name__ == '__main__':
    """
    qmix 1
    ensemble:
    5 1
    10 2
    15 0 good
    averaged:
    5 2 good
    10 1
    15 2
    """
    which = 'Ensemble'
    # which = 'Averaged'
    # plot_one_run_compare(which, how_many=50)
    plot_multi_run_compare(k=[15,5], maps='5m_vs_6m', num=3, how_many=100)
    # plot_reward_compare(which=which, maps='5m_vs_6m', n_run=[1,2,1,2], how_many=100, show=False)

    # 如果已经有图片就删掉
    # for filename in os.listdir(args.result_dir):
    #     if filename.endswith('.png'):
    #         os.remove(args.result_dir + '/' + filename)
    # plt.savefig(args.result_dir + "/%s.png" % tag)

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

    # plt.close()