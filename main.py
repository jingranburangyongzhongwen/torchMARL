# -*- coding: utf-8 -*-
"""
@Auth ： Kunfeng Li
@IDE ：PyCharm
"""
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_q_decom_args
from common.runner import Runner
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
import os
from envs.matrix_game_2 import Matrix_game2Env
from envs.matrix_game_3 import Matrix_game3Env
from envs.mmdp_game_1 import mmdp_game1Env
from envs.uni_mmdp import uni_mmdp_Env
from envs.env_GoOrderly import EnvGoOrderly
import torch
import numpy as np
import re


def main(env, arg, itr, seed=None):
    runner = Runner(env, arg, itr, seed)
    # print(runner.get_initial_qtot())
    # 如果训练模型
    if arguments.learn:
        runner.run()
        # runner.run_steps()
    runner.save_results()
    runner.plot()


def get_env(arg):
    if arguments.map == 'matrix_2':
        # 210000
        return Matrix_game2Env()
    elif arguments.map == 'matrix_3':
        return Matrix_game3Env(n_agents=2,
                               n_actions=3,
                               episode_limit=1,
                               obs_last_action=False,
                               state_last_action=False,
                               print_rew=False,
                               is_print=False)
    elif 'mmdp-' in arguments.map:
        length = int(re.findall(r'\d+\.\d+|\d+', arg.map)[-1])
        return uni_mmdp_Env(episode_limit=length)
    elif arguments.map == 'go_orderly':
        return EnvGoOrderly(map_size=6, num_agent=3)
    else:
        # 设置环境，pymarl中设置的也是环境默认参数
        return StarCraft2Env(map_name=arg.map,
                             difficulty=arg.difficulty,
                             step_mul=arg.step_mul,
                             replay_dir=arg.replay_dir)


def get_info(env, arg):
    env_info = env.get_env_info()
    # 动作个数
    arg.n_actions = env_info['n_actions']
    # agent数目
    arg.n_agents = env_info['n_agents']
    # 状态空间size
    arg.state_shape = env_info['state_shape']
    # 观察空间size
    arg.obs_shape = env_info['obs_shape']
    # her 的观测包含目标
    if arg.her:
        arg.obs_shape += 2
    # episode长度限制
    arg.episode_limit = env_info['episode_limit']
    # 获取单位维度
    if 'mmdp' in arg.map:
        arg.unit_dim = env.unit_dim
    elif not 'matrix' in arg.map and \
            not 'go_orderly' in arg.map:
        arg.unit_dim = 4 + env.shield_bits_ally + env.unit_type_bits
    return arg


if __name__ == '__main__':
    start = time.time()
    # start_time = time.strftime("%Y-%m-%d_%H-%M-%S", )
    start_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
    print('开始时间：' + start_time)
    arguments = get_common_args()

    if 'mmdp' in arguments.map:
        arguments.min_epsilon = 1
        arguments.target_update_period = 1
        arguments.didactic = True
        # 600
        arguments.max_steps = 60000
        arguments.evaluate_num = 32
        arguments.bs_rate = 1

    arguments = get_q_decom_args(arguments)

    if arguments.didactic:
        arguments.evaluation_steps_period = 500
        arguments.save_model_period = arguments.max_steps // 10

    if arguments.gpu is not None:
        arguments.cuda = True
        arguments.device = torch.device('cuda')
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
        # if arguments.gpu == 'a':
        #     pass
        # else:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        arguments.cuda = False
        arguments.device = torch.device('cpu')

    environment = get_env(arguments)

    # 获取环境信息
    arguments = get_info(environment, arguments)

    print('min_epsilon: {}'.format(arguments.min_epsilon))
    print('地图: {}'.format(arguments.map))

    # 进程池，数字是并行的进程数，根据资源自行调整，默认是CPU核的个数
    if arguments.num > 1:
        p = Pool(12)
        for i in range(arguments.num):
            if arguments.didactic:
                seed = int(time.time() % 10000000)
            else:
                seed = i
            p.apply_async(main, args=(environment, arguments, str(i) + '-' + start_time, seed))
            time.sleep(5)
        print('子进程开始...')
        p.close()
        p.join()
        print('所有子进程结束！')
    else:
        if arguments.didactic:
            seed = None
        else:
            seed = 0
        main(environment, arguments, start_time, seed)

    duration = time.time() - start
    time_list = [0, 0, 0]
    time_list[0] = duration // 3600
    time_list[1] = (duration % 3600) // 60
    time_list[2] = round(duration % 60, 2)
    print('用时：' + str(time_list[0]) + ' 时 ' + str(time_list[1]) + '分' + str(time_list[2]) + '秒')
    print('开始时间：' + start_time)
    end_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
    print('结束时间：' + end_time)
