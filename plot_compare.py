# -*- coding: utf-8 -*-
"""
@Auth ： Kunfeng Li
@IDE ：PyCharm
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
import pandas as pd
import math
from copy import deepcopy


def load_results(pre, alg=None, i=None):
    """
    从指定位置加载数据，方便后续作图
    """
    path = pre + '/' + alg + '/' + str(i)
    return np.load(path + '/evaluate_itr.npy'), np.load(path + '/win_rates.npy'), \
           np.load(path + '/episodes_rewards.npy')


def plot_multi_run_compare(pre, algs, num=5, mode=0, s=0, window=10, split=None, unit='mil', how_many=None,
                           show=True, pic_name=None):
    """
    绘制多个方法的多次运行数据
    """
    # colors = ['blue', 'k', 'r', 'tan', 'yellowgreen', 'darksage', 'g', 'c', 'purple']
    win_compare = None
    win_mean = None
    reward_compare = None
    reward_var = None
    reward_roll = None

    standard_itr = None
    for alg in algs:
        for i in range(num):
            evaluate_itr, win_rates, episodes_rewards = load_results(pre, alg, i)
            if unit is 'mil':
                evaluate_itr = evaluate_itr / 1000000.
            else:
                evaluate_itr = evaluate_itr / 1000.
            # 数据对齐
            if unit is 'mil':
                while win_rates.shape[0] < 299:
                    win_rates = np.append(win_rates, win_rates[-1])
                while evaluate_itr.shape[0] < 299:
                    if evaluate_itr[-1] <= 1.:
                        evaluate_itr = np.append(evaluate_itr, 1.)
                    else:
                        evaluate_itr = np.append(evaluate_itr, 2.)
                while episodes_rewards.shape[0] < 299:
                    episodes_rewards = np.concatenate([episodes_rewards, episodes_rewards[-1][None, :]], axis=0)

            if how_many is not None:
                interval = math.ceil(evaluate_itr.shape[0] / how_many)
                indexes = np.arange(0, evaluate_itr.shape[0], interval, dtype=np.int)

                evaluate_itr = evaluate_itr[indexes]
                win_rates = win_rates[indexes]
                episodes_rewards = episodes_rewards[indexes]
            if standard_itr is None:
                standard_itr = evaluate_itr

            alg_tag = np.full(standard_itr.shape, fill_value=alg)
            tmp_win = pd.DataFrame(np.concatenate((standard_itr[:, None], alg_tag[:, None], win_rates[:, None]),
                                                  axis=1),
                                   columns=['Steps ('+unit+')', 'Methods', 'Median Test Win'])
            tmp_win_mean = deepcopy(tmp_win)
            tmp_win_mean['Median Test Win'] = tmp_win['Median Test Win']. \
                rolling(window=window, min_periods=None).mean()
            if win_compare is None:
                win_compare = tmp_win
                win_mean = tmp_win_mean
            else:
                win_compare = pd.concat([win_compare, tmp_win], axis=0)
                win_mean = pd.concat([win_mean, tmp_win_mean], axis=0)

            tmp_reward = pd.DataFrame(np.concatenate((standard_itr[:, None], alg_tag[:, None],
                                                      np.median(episodes_rewards, axis=1)[:, None]), axis=1),
                                      columns=['Steps ('+unit+')', 'Methods', 'Median Joint-action Value'])

            tmp_reward_roll = deepcopy(tmp_reward)
            tmp_reward_var = deepcopy(tmp_reward)
            if s == 0:
                tmp_reward_var['Median Joint-action Value'] = tmp_reward['Median Joint-action Value'].\
                    rolling(window=window, min_periods=None).kurt()
                tmp_reward_roll['Median Joint-action Value'] = tmp_reward['Median Joint-action Value']. \
                    rolling(window=window, min_periods=None).mean()
            else:
                tmp_reward_var['Median Joint-action Value'] = tmp_reward['Median Joint-action Value']. \
                    rolling(window=window, min_periods=None).skew()
                tmp_reward_roll['Median Joint-action Value'] = tmp_reward['Median Joint-action Value']. \
                    rolling(window=window, min_periods=None).std()

            tmp_reward_var = tmp_reward_var[window:]
            tmp_reward_roll = tmp_reward_roll[window:]
            if reward_compare is None:
                reward_compare = tmp_reward
                reward_var = tmp_reward_var
                reward_roll = tmp_reward_roll
            else:
                reward_compare = pd.concat([reward_compare, tmp_reward], axis=0)
                reward_var = pd.concat([reward_var, tmp_reward_var], axis=0)
                reward_roll = pd.concat([reward_roll, tmp_reward_roll], axis=0)

    win_compare[['Steps ('+unit+')', 'Median Test Win']] = \
        win_compare[['Steps ('+unit+')', 'Median Test Win']].apply(pd.to_numeric)
    win_mean[['Steps (' + unit + ')', 'Median Test Win']] = \
        win_mean[['Steps (' + unit + ')', 'Median Test Win']].apply(pd.to_numeric)
    reward_compare[['Steps ('+unit+')', 'Median Joint-action Value']] = \
        reward_compare[['Steps ('+unit+')', 'Median Joint-action Value']].apply(pd.to_numeric)
    reward_var[['Steps ('+unit+')', 'Median Joint-action Value']] = \
        reward_var[['Steps ('+unit+')', 'Median Joint-action Value']].apply(pd.to_numeric)
    reward_roll[['Steps ('+unit+')', 'Median Joint-action Value']] = \
        reward_roll[['Steps ('+unit+')', 'Median Joint-action Value']].apply(pd.to_numeric)

    sns.set(context='paper', style='white')
    fig, ax = plt.subplots()

    # 之前做mmdp实验是联合动作值，其余实验就是return
    if mode == 0:
        if unit is 'k':
            y_name = 'Median Joint-action Value'
        else:
            y_name = 'Median Test Return'
        data = reward_compare.rename(columns=
                                     {'Median Joint-action Value': y_name})
    elif mode == 1:
        y_name = 'Median Skewness'
        if s == 0:
            y_name = 'Median Kurtosis'
            # plt.hlines(y=-3, xmin=min(standard_itr), xmax=max(standard_itr), colors="b", linestyles="dashed")

        data = reward_var.rename(columns={'Median Joint-action Value': y_name})
    elif mode == 2:
        y_name = 'Collective Returns' # Rolling Window Mean
        if s == 1:
            y_name = 'Median Joint-action Value std'
        data = reward_roll.rename(columns=
                                  {'Median Joint-action Value': y_name})
    elif mode == 3:
        plt.ylim(0, 1.0)
        data = win_compare
        y_name = 'Median Test Win'
    elif mode == 4:
        plt.ylim(0, 1.0)
        data = win_mean
        y_name = 'Median Test Win'
    else:
        raise Exception('没有这个mode！')

    # plt.title(map)
    if split is None:
        sns.lineplot(x='Steps (' + unit + ')', y=y_name, hue='Methods',
                     data=data,
                     ci=68, estimator=np.median)
    else:
        data_dict = dict(list(data.groupby('Methods')))
        sns.lineplot(x='Steps (' + unit + ')', y=y_name, hue='Methods',
                     data=data_dict[algs[0]], ax=ax,
                     ci=68, estimator=np.median)
        index = 0
        colors = [1, 2, 3, 4, 5, 6]
        # sns.color_palette("Paired")[index]
        for i in range(1, len(algs)):
            sns.lineplot(x='Steps (' + unit + ')', y=y_name, color=sns.color_palette("colorblind")[colors[index]],
                         data=data_dict[algs[i]], label=algs[i], ax=ax,
                         ci=68, estimator=np.median)
            if type(split) is list:
                if split[0] in algs[i]:
                    ax.lines[i + 2].set_linestyle("--")
                if split[-1] in algs[i]:
                    index += 1
                    ax.lines[i + 2].set_linestyle((0, (1, 1)))
            else:
                if split in algs[i]:
                    index += 1
                    ax.lines[i + 2].set_linestyle("--")
    plt.legend()
    # 格式化成2020-03-20_11-45-39形式
    tag = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    if show:
        plt.show()
    else:
        if pic_name:
            plt.savefig(pic_name+'.pdf', dpi=600, format='pdf', bbox_inches='tight')
        else:
            plt.savefig('pic.pdf', dpi=600, format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    plot_multi_run_compare(pre='./results-steps/go_orderly',
                           unit='mil',  # x轴的单位，k或者mil，mmdp要改成k
                           algs=['QMIX-HER=3', 'QMIX'],
                           mode=4, # 0在MMDP实验中是Qtot，否则是return
                                   # 1是峰度或偏度
                                   # 2是reward rolling mean或者std
                                   # 3是胜率
                                   # 4是胜率 rolling mean或者std
                           s=0, # 0是峰度，1是偏度，0是mean，1是std
                           num=5,
                           window=10, # rolling的窗口大小
                           # split=['adapt', 'eval'], # 算法名字里有这个关键字就将其变成虚线
                           how_many=None,
                           show=False,
                           pic_name='win_rates')

