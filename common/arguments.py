# -*- coding: utf-8 -*-
"""
@Time ： 2020/7/15 17:05
@Auth ： Kunfeng Li
@File ：arguments.py
@IDE ：PyCharm

"""
import argparse


def get_common_args():
    """
    得到通用参数
    :return:
    """
    parser = argparse.ArgumentParser()
    # 星际争霸环境设置
    parser.add_argument('--map', type=str, default='5m_vs_6m', help='使用的地图')
    parser.add_argument('--difficulty', type=str, default='7', help='游戏难度')
    parser.add_argument('--game_version', type=str, default='latest', help='游戏版本')
    parser.add_argument('--seed', type=int, default=123, help='随机数种子')
    parser.add_argument('--step_mul', type=int, default=8, help='多少步执行动作')
    parser.add_argument('--replay_dir', type=str, default='', help='回放的绝对路径')
    # 算法设置
    parser.add_argument('--alg', type=str, default='qmix', help='选择所使用的算法')
    parser.add_argument('--last_action', type=bool, default=True, help='是否使用上一个动作帮助决策')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--optim', type=str, default='RMS', help='优化器')
    parser.add_argument('--evaluate_num', type=int, default=20, help='测试的次数')
    parser.add_argument('--reuse_network', type=bool, default=True, help='是否共享一个网络')
    parser.add_argument('--k', type=int, default=1, help='使用多少个进行ensemble或者average')
    # 程序运行设置
    parser.add_argument('--result_dir', type=str, default='./results', help='保存模型和结果的位置')
    parser.add_argument('--model_dir', type=str, default='./model', help='这个策略模型的地址')
    parser.add_argument('--load_model', type=bool, default=False, help='是否加载已有模型')
    parser.add_argument('--learn', type=bool, default=True, help='是否训练模型')
    parser.add_argument('--gpu', type=str, default=None, help='使用哪个GPU，默认不使用')
    parser.add_argument('--num', type=int, default=1, help='并行执行多少个程序进程')

    args = parser.parse_args()
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
    # 最大迭代次数
    args.n_itr = 2000000
    # 一个itr里有多少个episode
    args.n_episodes = 1
    # 一个 itr 里训练多少次
    args.train_steps = 1
    # 多久评价一次
    args.evaluation_period = 100
    # 经验池
    args.batch_size = 32
    args.buffer_size = int(5e3)
    # 模型保存周期
    args.save_model_period = 5000
    # target网络更新周期
    args.target_update_period = 200
    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1
    # 梯度裁剪
    args.clip_norm = 10
    # maven
    return args


