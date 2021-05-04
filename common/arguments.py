# -*- coding: utf-8 -*-
"""
@Auth ： Kunfeng Li
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
    parser.add_argument('--map', type=str, default='go_orderly', help='使用的地图')
    # 指定算法 qplex 就是dmaq
    parser.add_argument('--alg', type=str, default='qmix', help='选择所使用的算法')
    parser.add_argument('--her', type=int, default=None, help='buffer的类型，是否使用her，这个值代表k')
    # 程序运行设置
    parser.add_argument('--gpu', type=str, default=None, help='使用哪个GPU，默认不使用')
    parser.add_argument('--num', type=int, default=1, help='并行执行多少个程序进程')
    # -------------------------------------这些参数一般不会更改-------------------------------------
    # 部分参数设置
    parser.add_argument('--target_update_period', type=int, default=200, help='target网络更新周期，episode')
    # parser.add_argument('--n_itr', type=int, default=20000, help='最大迭代次数') TODO 2000000 500000
    parser.add_argument('--max_steps', type=int, default=2000000, help='最大迭代time steps')
    parser.add_argument('--bs_rate', type=float, default=12.5, help='replay buffer是batch size的多少倍时开始训练')
    # 算法设置
    parser.add_argument('--optim', type=str, default='RMS', help='优化器')
    # 程序运行设置
    parser.add_argument('--result_dir', type=str, default='./results-steps', help='保存模型和结果的位置')
    parser.add_argument('--model_dir', type=str, default='./model-steps', help='这个策略模型的地址')
    parser.add_argument('--load_model', type=bool, default=False, help='是否加载已有模型')
    parser.add_argument('--learn', type=bool, default=True, help='是否训练模型')
    parser.add_argument('--epsilon_anneal_steps', type=int, default=50000, help='epsilon 经过多少降低到最小')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='最小epsilon')

    args = parser.parse_args()

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
    # 是否使用上一个动作帮助决策
    args.last_action = True
    # 是否共享一个网络
    args.reuse_network = True
    # 折扣因子
    args.gamma = 0.99
    # 测试的次数
    args.evaluate_num = 32
    args.didactic = False
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
    # qtran "qtran_base"
    if 'qtran' in args.alg:
        args.qtran_hidden_dim = 64
        args.qtran_arch = "qtran_paper"
        args.mixing_embed_dim = 64
        args.opt_loss = 1
        args.nopt_min_loss = 0.1
        args.network_size = 'small'
        # QTRAN lambda
        args.lambda_opt = 1
        args.lambda_nopt = 1
    # qatten
    elif args.alg == 'qatten':
        args.mixing_embed_dim = 32
        args.n_head = 4  # attention head number
        args.attend_reg_coef = 0.001  # attention regulation coefficient  # For MMM2 and 3s5z_vs_3s6z, it is 0.001
        args.hypernet_layers = 2
        args.nonlinear = False  # non-linearity, for MMM2, it is True
        args.weighted_head = False  # weighted head Q-values, for MMM2 and 3s5z_vs_3s6z, it is True
        args.state_bias = True
        args.hypernet_embed = 64
        args.mask_dead = False
    elif 'dmaq' in args.alg:
        args.mixing_embed_dim = 32
        args.hypernet_embed = 64
        args.adv_hypernet_layers = 3
        args.adv_hypernet_embed = 64

        args.num_kernel = 10
        args.is_minus_one = True
        args.weighted_head = True
        args.is_adv_attention = True
        args.is_stop_gradient = True
    elif 'wqmix' in args.alg:
        args.central_loss = 1
        args.qmix_loss = 1
        args.w = 0.1  # $\alpha$ in the paper
        # False -> CW-QMIX, True -> OW-QMIX
        args.hysteretic_qmix = True
        if args.alg == 'cwqmix':
            args.hysteretic_qmix = False

        args.central_mixing_embed_dim = 256
        args.central_action_embed = 1
        args.central_mac = "basic_central_mac"
        args.central_agent = "central_rnn"
        args.central_rnn_hidden_dim = 64
        args.central_mixer = "ff"

    # 学习率
    args.lr = 5e-4
    # epsilon greedy
    args.epsilon = 1
    args.epsilon_decay = (args.epsilon - args.min_epsilon) / args.epsilon_anneal_steps
    args.epsilon_anneal_scale = 'step'
    # 一个itr里有多少个episode
    args.n_episodes = 1
    # 一个 itr 里训练多少次
    args.train_steps = 1
    # 多久评价一次，pymarl中产生200个点，即循环两百万次，每一万次开始评价
    # args.evaluation_period = 100
    # TODO step 相关
    args.evaluation_steps_period = math.ceil(args.max_steps / 300.)
    # args.evaluation_steps_period = 10000
    # 经验池，采样32个episode
    args.batch_size = 32
    args.buffer_size = int(5e3)
    # 模型保存周期
    # args.save_model_period = 5000
    # args.save_model_period = math.ceil(args.n_itr / 200.)
    args.save_model_period = math.ceil(args.max_steps / 100.)
    # 梯度裁剪
    args.clip_norm = 10
    return args
