# -*- coding: utf-8 -*-
"""
@Time ： 2020/7/15 17:05
@Auth ： Kunfeng Li
@File ：arguments.py
@IDE ：PyCharm

"""
import argparse
import re
import math


def get_common_args():
    """
    得到通用参数
    :return:
    """
    parser = argparse.ArgumentParser()
    # 星际争霸环境设置
    parser.add_argument('--map', type=str, default='5m_vs_6m', help='使用的地图')
    # 算法设置 ensemble
    parser.add_argument('--alg', type=str, default='cwqmix', help='选择所使用的算法')
    parser.add_argument('--last_action', type=bool, default=True, help='是否使用上一个动作帮助决策')
    parser.add_argument('--optim', type=str, default='RMS', help='优化器')
    parser.add_argument('--reuse_network', type=bool, default=True, help='是否共享一个网络')
    # 程序运行设置
    parser.add_argument('--result_dir', type=str, default='./results', help='保存模型和结果的位置')
    parser.add_argument('--model_dir', type=str, default='./model', help='这个策略模型的地址')
    parser.add_argument('--load_model', type=bool, default=False, help='是否加载已有模型')
    parser.add_argument('--learn', type=bool, default=True, help='是否训练模型')
    parser.add_argument('--gpu', type=str, default=None, help='使用哪个GPU，默认不使用')
    parser.add_argument('--num', type=int, default=1, help='并行执行多少个程序进程')
    # 部分参数设置
    parser.add_argument('--n_itr', type=int, default=200000, help='最大迭代次数')

    args = parser.parse_args()
    # -------------------------------------这些参数一般不会更改-------------------------------------
    # 游戏难度
    args.difficulty = '7'
    # 多少步执行动作
    args.step_mul = 8
    # 游戏版本
    # args.game_version = 'latest'
    # 随机数种子
    args.seed = 123
    # 回放的绝对路径
    args.replay_dir = ''
    # 折扣因子
    args.gamma = 0.99
    # 测试的次数
    args.evaluate_num = 32
    return args


def get_q_decom_args(args):
    """
    得到值分解算法（vdn, qmix, qtran）的参数
    :param args:
    :return:
    """
    # 网络设置
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    # 学习率
    args.lr = 5e-4
    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    args.epsilon_decay = (args.epsilon - args.min_epsilon) / 50000
    args.epsilon_anneal_scale = 'step'
    # 一个itr里有多少个episode
    args.n_episodes = 1
    # 一个 itr 里训练多少次
    args.train_steps = 1
    # TODO 多久评价一次，pymarl中产生200个点，即循环两百万次，每一万次开始评价，所以最好还是确定一个迭代次数后更改这个
    # args.evaluation_period = 100
    args.evaluation_period = math.ceil(args.n_itr / 300.)
    # 经验池，采样32个episode
    args.batch_size = 32
    args.buffer_size = int(5e3)
    # 模型保存周期
    # args.save_model_period = 5000
    args.save_model_period = math.ceil(args.n_itr / 200.)
    # target网络更新周期，episode
    args.target_update_period = 200
    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1
    # 梯度裁剪
    args.clip_norm = 10
    # maven
    return args


