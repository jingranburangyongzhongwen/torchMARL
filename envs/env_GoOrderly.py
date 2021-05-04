# -*- coding: utf-8 -*-
"""
@Auth ： Kunfeng Li
@IDE ：PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from collections import Counter
# import cv2
import copy


class EnvGoOrderly(object):
    def __init__(self, map_size, num_agent):
        self.map_size = map_size
        self.num_agent = num_agent
        self.state = []
        self.goal  = []
        self.order = [[0]*num_agent for _ in range(num_agent)]
        self.former_states = []
        self.occupancy = np.zeros((self.map_size, self.map_size))
        self._episode_steps = 0
        self._episode_limit = map_size*map_size*2

        # 产生初始状态
        # for i in range(self.num_agent):
        #     goal = [random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2)]
        #     while goal in self.state or goal in self.goal:
        #         goal = [random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2)]
        #     self.goal.append(goal)

    def reset(self):
        self._episode_steps = 0
        self.state = []
        self.goal = []
        # self.order = []
        self.former_states = []
        self.occupancy = np.zeros((self.map_size, self.map_size))
        # 产生先后次序
        # order = [random.randint(1, self.num_agent) for _ in range(self.num_agent)]
        # # 针对顺序的重新编码，如果order=[2,2,3]，则改为[1,1,2]
        # order_count = dict(Counter(order))
        # if len(order_count) < self.num_agent:
        #     for i in range(len(order_count)):
        #         ifexist = False
        #         for j in range(self.num_agent):
        #             if order[j] == (i + 1):
        #                 ifexist = True
        #         if not ifexist:
        #             minmax = 100
        #             for j in range(self.num_agent):
        #                 if order[j] > (i + 1) and order[j] < minmax:
        #                     minmax = order[j]
        #             for j in range(self.num_agent):
        #                 if order[j] == minmax:
        #                     order[j] = (i + 1)
        # for i in range(self.num_agent):
        #     o_=[]
        #     for j in range(self.num_agent):
        #         if order[i] == j+1:
        #             o_.append(1)
        #         else:
        #             o_.append(0)
        #     self.order.append(o_)

        # 产生初始状态
        for i in range(self.num_agent):
            # 随机生成初始状态和目标状态，且二者不能相等
            state = [random.randint(1, self.map_size-2), random.randint(1, self.map_size-2)]
            while state in self.state or state in self.goal:
                state = [random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2)]
            self.state.append(state)
            goal  = [random.randint(1, self.map_size-2), random.randint(1, self.map_size-2)]
            while goal in self.state or goal in self.goal:
                goal = [random.randint(1, self.map_size-2), random.randint(1, self.map_size-2)]
            self.goal.append(goal)

        # 产生初始前继状态
        # num_former_states = 0
        # #print(self.order)
        # for i in range(self.num_agent):
        #     idx = self.order[i].index(1)
        #     if idx > num_former_states:
        #         num_former_states = idx
        # for i in range(num_former_states + 1):
        #     self.former_states.append([])
        #     self.former_states[i] = copy.deepcopy(self.state)

        for i in range(self.map_size):
            self.occupancy[0][i] = 1
            self.occupancy[self.map_size - 1][i] = 1
            self.occupancy[i][0] = 1
            self.occupancy[i][self.map_size - 1] = 1

        # 打印每个agent的目标状态
        #print('***********************************************')
        #for i in range(self.num_agent):
        #    print('* Agent', i, ' Goal :', self.goal[i], ' Order:', self.order[i], '*')
        #    print('* Agent', i, ' State:', self.state[i], ' Dis:  ', 123)
        #print('***********************************************')

        # self.occupancy = np.zeros((self.map_size, self.map_size))
        # for i in range(self.map_size):
        #     self.occupancy[0][i] = 1
        #     self.occupancy[self.map_size - 1][i] = 1
        #     self.occupancy[i][0] = 1
        #     self.occupancy[i][self.map_size - 1] = 1
        # self.agt1_pos = [self.map_size - 3, 1]
        # self.agt2_pos = [self.map_size - 2, 2]
        # self.goal1_pos = [random.randint(2, self.map_size - 2), random.randint(2, self.map_size - 2)]
        # self.goal2_pos = [random.randint(2, self.map_size - 2), random.randint(2, self.map_size - 2)]

    def get_env_info(self):
        env_info = {"state_shape": (4 + self.num_agent) * self.num_agent,
                    "obs_shape": 4 + self.num_agent,
                    "n_actions": 5,
                    "n_agents": self.num_agent,
                    "episode_limit": self._episode_limit,
                    "unit_dim": 2}
        return env_info

    def get_state0(self):
        state = np.zeros((self.num_agent, 4))
        for i in range(self.num_agent):
            state[i,0] = self.state[i][0] / self.map_size
            state[i,1] = self.state[i][1] / self.map_size
            state[i,2] = self.goal[i][0] / self.map_size
            state[i,3] = self.goal[i][1] / self.map_size
        return state

    def get_state(self):
        state = np.zeros((1, (4+self.num_agent)*self.num_agent))
        for i in range(self.num_agent):
            for j in range(4+self.num_agent):
                if j == 0:
                    state[0,(4+self.num_agent)*i+0] = self.state[i][0] / (self.map_size-2)
                elif j == 1:
                    state[0,(4+self.num_agent)*i+1] = self.state[i][1] / (self.map_size-2)
                elif j == 2:
                    state[0,(4+self.num_agent)*i+2] = self.goal[i][0] / (self.map_size-2)
                elif j == 3:
                    state[0,(4+self.num_agent)*i+3] = self.goal[i][1] / (self.map_size-2)
                else:
                    state[0,(4+self.num_agent)*i+j] = self.order[i][j-4]
        # print(state[0])
        return state[0]

    def get_obs(self):
        obs = np.zeros((self.num_agent, 4+self.num_agent))
        for i in range(self.num_agent):
            obs[i, 0] = self.state[i][0] / (self.map_size-2)
            obs[i, 1] = self.state[i][1] / (self.map_size-2)
            obs[i, 2] = self.goal[i][0] / (self.map_size-2)
            obs[i, 3] = self.goal[i][1] / (self.map_size-2)
            for j in range(self.num_agent):
                obs[i, 4+j] = self.order[i][j]
        return obs

    def get_reward(self, state, former_states):
        '''
        如果有agent到达目的，则负的reward/num，越多奖赏越高，全到了就是0
        '''
        reward = -1
        reward_flag = True
        num = 0
        for i in range(self.num_agent):
            if state[i] == self.goal[i]:
                num += 1
        if num == self.num_agent:
            return 0
        elif num > 0:
            return float(reward)/num

        return reward

    def get_reward_old(self, state, former_states):
        reward = -1
        # 根据到达次序给定奖赏，只有完全满足order次序到达才能给奖赏
        reward_flag = True
        num = 0
        for i in range(self.num_agent):
            if state[i] == self.goal[i]:
                num += 1
        if num != self.num_agent:
            reward_flag = False
        else:
            #print('当前全部到达')
            for i in range(self.num_agent):
                idx = self.order[i].index(1)
                consist = len(former_states) - (idx+1)
                # 判断其到达目标后是不是一直保持住了，没判断其之前是不是已经到了，所以某一个提前到了也是可以的
                for j in range(consist):
                    if former_states[j][i] != self.goal[i]:
                        reward_flag = False
                if consist == 0:
                    if former_states[0][i] == self.goal[i]:
                        reward_flag = False

        if reward_flag:
            reward = 0

        return reward

    def step(self, action_list):
        state = copy.deepcopy(self.state)
        '''
        goal是各个agent目标的坐标
        order是顺序的one-hot编码
        当所有agent到达各自的目标，并且顺序符合order，则有reward
        每个agent到达对应的目标后可移动，就要前后到达才行
        former_states就是记录最近的历史，但其实如果
        '''

        for i in range(self.num_agent):
            # agent_i move
            if action_list[i] == 0:  # move up
                if self.occupancy[self.state[i][0] - 1][self.state[i][1]] != 1:  # if can move
                    self.state[i][0] = self.state[i][0] - 1
            elif action_list[i] == 1:  # move down
                if self.occupancy[self.state[i][0] + 1][self.state[i][1]] != 1:  # if can move
                    self.state[i][0] = self.state[i][0] + 1
            elif action_list[i] == 2:  # move left
                if self.occupancy[self.state[i][0]][self.state[i][1] - 1] != 1:  # if can move
                    self.state[i][1] = self.state[i][1] - 1
            elif action_list[i] == 3:  # move right
                if self.occupancy[self.state[i][0]][self.state[i][1] + 1] != 1:  # if can move
                    self.state[i][1] = self.state[i][1] + 1
            elif action_list[i] == 4:  # stay
                pass
        # 最后的在第一位
        if len(self.former_states) > 0:
            for i in range(len(self.former_states)-1,0,-1):
                self.former_states[i] = copy.deepcopy(self.former_states[i-1])
            self.former_states[0] = copy.deepcopy(state)

        # if self.state[0] == self.goal[0] and self.state[1] == self.goal[1]:
        #     reward = reward + 10
        #
        # if self.sqr_dist(self.state[0], self.goal[0])<=1 or self.sqr_dist(self.state[1], self.goal[1])>9:
        #     reward = reward - 0.5
        reward = self.get_reward(self.state, self.former_states)
        # if self.state[0] == self.goal[0] and self.state[1] == self.goal[1]:
        #     reward = 0
        self._episode_steps += 1
        info = {'battle_won': False}
        done = False
        if reward >= 0:
            reward = 0
            done = True
            info['battle_won'] = True
        elif self._episode_steps >= self._episode_limit:
            done = True
        return reward, done, info

    def sqr_dist(self, pos1, pos2):
        return (pos1[0]-pos2[0])*(pos1[0]-pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1])

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 4))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                    obs[i, j, 3] = 1.0
        for i in range(self.num_agent):
            if i%6 == 0:
                # 分第一组
                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            elif i%6 == 1:
                # 第二组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 1.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            elif i%6 == 2:
                #第三组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 1.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0
            elif i%6 == 3:
                #第四组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0
            elif i%6 == 4:
                #第五组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 1.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            else:
                #第六组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 1.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0

        return obs

    def plot_scene(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.get_global_obs())
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # def render(self):
    #     obs = self.get_global_obs()
    #     enlarge = 40
    #     henlarge = int(enlarge/2)
    #     qenlarge = int(enlarge/8)
    #     new_obs = np.ones((self.map_size*enlarge, self.map_size*enlarge, 3))
    #     for i in range(self.map_size):
    #         for j in range(self.map_size):
    #             if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
    #                 cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
    #             # 红色方形agent及其红色圆形目标
    #             if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
    #                 cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 255), -1)
    #             if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
    #                 cv2.circle(new_obs, ((2*j+1) * henlarge, (2*i+1) * henlarge), henlarge, (0, 0, 255),-1)
    #                 order = str(self.order[0].index(1) + 1)
    #                 cv2.putText(new_obs, order, ((8*j+3) * qenlarge, (8*i+5)*qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #             # 绿色方形agent及其绿色圆形目标
    #             if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
    #                 cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 0), -1)
    #             if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
    #                 cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 255, 0), -1)
    #                 order = str(self.order[1].index(1) + 1)
    #                 cv2.putText(new_obs, order, ((8*j+3) * qenlarge, (8*i+5)*qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #             # 蓝色方形agent及其蓝色圆形目标
    #             if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
    #                 cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 0), -1)
    #             if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 1.0:
    #                 cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 0, 0), -1)
    #                 order = str(self.order[2].index(1) + 1)
    #                 cv2.putText(new_obs, order, ((8*j+3) * qenlarge, (8*i+5)*qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #             # 青色方形agent及其青色圆形目标
    #             if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
    #                 cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 255, 0), -1)
    #             if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
    #                 cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 255, 0), -1)
    #                 order = str(self.order[3].index(1) + 1)
    #                 cv2.putText(new_obs, order, ((8 * j + 3) * qenlarge, (8 * i + 5) * qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #             # 黄色方形agent及其黄色圆形目标
    #             if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
    #                 cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 255), -1)
    #             if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
    #                 cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 255, 255), -1)
    #                 order = str(self.order[4].index(1) + 1)
    #                 cv2.putText(new_obs, order, ((8 * j + 3) * qenlarge, (8 * i + 5) * qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #             # 粉色方形agent及其粉色圆形目标
    #             if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
    #                 cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 255), -1)
    #             if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 1.0:
    #                 cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 0, 255), -1)
    #                 order = str(self.order[5].index(1) + 1)
    #                 cv2.putText(new_obs, order, ((8 * j + 3) * qenlarge, (8 * i + 5) * qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #
    #     cv2.imshow('image', new_obs)
    #     cv2.waitKey(100)

    def get_avail_agent_actions(self, agent_id):
        return [1] * 5

    def close(self):
        pass
