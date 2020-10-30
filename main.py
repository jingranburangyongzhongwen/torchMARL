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
from multiprocessing import Pool


def main(env, arg, itr):
    runner = Runner(env, arg, itr)
    # 如果训练模型
    if arguments.learn:
        runner.run()
    runner.save_results()
    runner.plot()


if __name__ == '__main__':
    start = time.time()
    arguments = get_q_decom_args(get_common_args())
    # 设置环境
    environment = StarCraft2Env(map_name=arguments.map,
                                difficulty=arguments.difficulty,
                                game_version=arguments.game_version,
                                step_mul=arguments.step_mul,
                                replay_dir=arguments.replay_dir)
    # 获取环境信息
    env_info = environment.get_env_info()
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

    # 进程池，数字是并行的进程数，根据资源自行调整，默认是CPU核的个数
    if arguments.num > 1:
        p = Pool(12)
        for i in range(arguments.num):
            p.apply_async(main, args=(environment, arguments, i))
        print('子进程开始...')
        p.close()
        p.join()
        print('所有子进程结束！')
    else:
        main(environment, arguments, 0)

    duration = time.time() - start
    time_list = [0, 0, 0]
    time_list[0] = duration // 3600
    time_list[1] = (duration % 3600) // 60
    time_list[2] = round(duration % 60, 2)
    print('用时：' + str(time_list[0]) + ' 时 ' + str(time_list[1]) + '分' + str(time_list[2]) + '秒')
    end_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('结束时间：' + end_time)
