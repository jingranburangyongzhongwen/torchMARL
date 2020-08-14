# -*- coding: utf-8 -*-
"""
@Time ： 2020/7/15 17:23
@Auth ： Kunfeng Li
@File ：main.py
@IDE ：PyCharm

"""
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_q_decom_args
from common.runner import Runner
import time


if __name__ == '__main__':
    start = time.time()
    arguments = get_q_decom_args(get_common_args())
    # 设置环境
    env = StarCraft2Env(map_name=arguments.map,
                        difficulty=arguments.difficulty,
                        game_version=arguments.game_version,
                        step_mul=arguments.step_mul,
                        replay_dir=arguments.replay_dir)
    # 获取环境信息
    env_info = env.get_env_info()
    # 动作个数
    arguments.n_actions = env_info['n_actions']
    # agent数目
    arguments.n_agents = env_info['n_agents']
    # 状态空间size
    arguments.state_shape = env_info['state_shape']
    # 观察空间size
    arguments.obs_shape = env_info['obs_shape']
    # episode长度限制
    arguments.episode_limit = env_info['episode_limit']

    runner = Runner(env, arguments)
    # 如果训练模型
    if arguments.learn:
        runner.run()
    runner.save_results()
    runner.plot()

    duration = time.time() - start
    time_list = [0, 0, 0]
    time_list[0] = duration // 3600
    time_list[1] = (duration % 3600) // 60
    time_list[2] = round(duration % 60, 2)
    print('用时：' + str(time_list[0]) + ' 时 ' + str(time_list[1]) + '分' + str(time_list[2]) + '秒')
    end_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('结束时间：' + end_time)
