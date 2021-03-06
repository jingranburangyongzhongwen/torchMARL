# -*- coding: utf-8 -*-
"""
@Auth ： Kunfeng Li
@IDE ：PyCharm
"""
import torch
import os
from network.base_net import RNN
from network.qmix_mixer import QMIXMixer
from network.vdn_mixer import VDNMixer
from network.wqmix_q_star import QStar
from network.qatten_mixer import QattenMixer
import numpy as np


class Q_Decom:
    def __init__(self, args, itr):
        self.args = args
        input_shape = self.args.obs_shape
        # 调整RNN输入维度
        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.args.n_agents
        # 设置网络
        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        # 通过数字标记，方便后续对算法类型进行判断
        self.wqmix = 0
        if self.args.alg == 'cwqmix' or self.args.alg == 'owqmix':
            self.wqmix = 1

        # 默认值分解算法使用QMIX
        if 'qmix' in self.args.alg:
            self.eval_mix_net = QMIXMixer(args)
            self.target_mix_net = QMIXMixer(args)
            if self.wqmix > 0:
                self.qstar_eval_mix = QStar(args)
                self.qstar_target_mix = QStar(args)
                self.qstar_eval_rnn = RNN(input_shape, args)
                self.qstar_target_rnn = RNN(input_shape, args)
                if self.args.alg == 'cwqmix':
                    self.alpha = 0.75
                elif self.args.alg == 'owqmix':
                    self.alpha = 0.5
                else:
                    raise Exception('没有这个算法')
        elif self.args.alg == 'vdn':
            self.eval_mix_net = VDNMixer()
            self.target_mix_net = VDNMixer()
        elif self.args.alg == 'qatten':
            self.eval_mix_net = QattenMixer(args)
            self.target_mix_net = QattenMixer(args)

        # 是否使用GPU
        if args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_mix_net.cuda()
            self.target_mix_net.cuda()
            if self.wqmix > 0:
                self.qstar_eval_mix.cuda()
                self.qstar_target_mix.cuda()
                self.qstar_eval_rnn.cuda()
                self.qstar_target_rnn.cuda()
        # 是否加载模型
        # self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/' + str(itr)
        # self.model_dir = args.model_dir + '/' + args.map + '/' + args.alg + '_' + str(self.args.epsilon_anneal_steps // 10000) + 'w' + '/' + str(itr)
        if args.load_model:
            if os.path.exists(self.args.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.args.model_dir + '/rnn_net_params.pkl'
                path_mix = self.args.model_dir + '/' + self.args.alg + '_net_params.pkl'

                map_location = 'cuda:0' if args.cuda else 'cpu'

                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_mix_net.load_state_dict(torch.load(path_mix, map_location=map_location))
                if self.wqmix > 0:
                    path_agent_rnn = self.args.model_dir + '/rnn_net_params2.pkl'
                    path_qstar = self.args.model_dir + '/' + 'qstar_net_params.pkl'
                    self.qstar_eval_rnn.load_state_dict(torch.load(path_agent_rnn, map_location=map_location))
                    self.qstar_eval_mix.load_state_dict(torch.load(path_qstar, map_location=map_location))
                print('成功加载模型 %s'%path_rnn + ' 和 %s'%path_mix)
            else:
                raise Exception("模型不存在")
        # 令target网络与eval网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        # 获取所有参数
        self.eval_params = list(self.eval_rnn.parameters()) + list(self.eval_mix_net.parameters())
        # 学习过程中要为每个episode的每个agent维护一个eval_hidden，执行过程中要为每个agetn维护一个eval_hidden
        self.eval_hidden = None
        self.target_hidden = None
        if self.wqmix > 0:
            # 令target网络与eval网络参数相同
            self.qstar_target_rnn.load_state_dict(self.qstar_eval_rnn.state_dict())
            self.qstar_target_mix.load_state_dict(self.qstar_eval_mix.state_dict())
            # 获取所有参数
            # self.qstar_params = list(self.qstar_eval_rnn.parameters()) + list(self.qstar_eval_mix.parameters())
            self.eval_params += list(self.qstar_eval_rnn.parameters()) + list(self.qstar_eval_mix.parameters())
            # init hidden
            self.qstar_eval_hidden = None
            self.qstar_target_hidden = None
        # 获取优化器
        if args.optim == 'RMS':
            self.optimizer = torch.optim.RMSprop(self.eval_params, lr=args.lr)
            # if self.wqmix > 0:
            #     self.qstar_optimizer = torch.optim.RMSprop(self.qstar_params, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_params)
            # if self.wqmix > 0:
            #     self.qstar_optimizer = torch.optim.Adam(self.qstar_params)
        print("值分解算法 " + self.args.alg + " 初始化")

    def learn(self, batch, max_episode_len, train_step):
        """
        在learn的时候，抽取到的数据是四维的，四个维度分别为
        1——第几个episode
        2——episode中第几个transition
        3——第几个agent的数据
        4——具体obs维度。
        因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，
        然后一次给神经网络传入每个episode的同一个位置的transition
        :param batch:
        :param max_episode_len:
        :param train_step:
        :param epsilon:
        :return:
        """
        # 获得episode的数目
        episode_num = batch['o'].shape[0]
        # 初始化隐藏状态
        self.init_hidden(episode_num)
        # 数据转为tensor
        # for key in batch.keys():
        #     if key == 'a':
        #         batch[key] = torch.LongTensor(batch[key])
        #     else:
        #         batch[key] = torch.Tensor(batch[key])
        for key in batch.keys():
            if key == 'a':
                batch[key] = torch.as_tensor(batch[key], dtype=torch.long, device=self.args.device)
            else:
                batch[key] = torch.as_tensor(batch[key], dtype=torch.float, device=self.args.device)

        s, next_s, a, r, avail_a, next_avail_a, done = batch['s'], batch['next_s'], batch['a'], \
                                                       batch['r'], batch['avail_a'], batch['next_avail_a'], \
                                                       batch['done']
        # 避免填充的产生 TD-error 影响训练
        mask = 1 - batch["padded"].float()
        # 获取当前与下个状态的q值，（episode, max_episode_len, n_agents, n_actions）
        eval_qs, target_qs = self.get_q(batch, episode_num, max_episode_len)
        # 是否使用GPU
        # if self.args.cuda:
        #     a = a.cuda()
        #     r = r.cuda()
        #     done = done.cuda()
        #     mask = mask.cuda()
        #     # if 'qmix' in self.args.alg:
        #     s = s.cuda()
        #     next_s = next_s.cuda()
        # 得到每个动作对应的 q 值
        eval_qsa = torch.gather(eval_qs, dim=3, index=a).squeeze(3)
        # 计算Q_tot
        if self.args.alg == 'qatten':
            eval_q_total, q_attend_regs, head_entropies = self.eval_mix_net(eval_qsa, s, a)
        else:
            eval_q_total = self.eval_mix_net(eval_qsa, s)
        qstar_q_total, qstar_loss, q_attend_regs = None, None, None
        # 需要先把不行动作的mask掉
        target_qs[next_avail_a == 0.0] = -9999999
        target_qsa = target_qs.max(dim=3)[0]
        if self.wqmix > 0:
            # TODO 找到使得Q_tot最大的联合动作，由于qmix是单调假设的，每个agent q值最大则 Q_tot最大，因此联合动作就是每个agent q值最大的动作
            argmax_u = target_qs.argmax(dim=3).unsqueeze(3)
            qstar_eval_qs, qstar_target_qs = self.get_q(batch, episode_num, max_episode_len, True)
            # 获得对应的动作q值
            qstar_eval_qs = torch.gather(qstar_eval_qs, dim=3, index=a).squeeze(3)
            qstar_target_qs = torch.gather(qstar_target_qs, dim=3, index=argmax_u).squeeze(3)
            # 通过前馈网络得到qstar
            qstar_q_total = self.qstar_eval_mix(qstar_eval_qs, s)
            next_q_total = self.qstar_target_mix(qstar_target_qs, next_s)
        elif self.args.alg == 'qatten':
            # chosen_action_qvals, q_attend_regs, head_entropies = self.mixer(chosen_action_qvals, batch["state"][:, :-1],
            #                                                                 actions)
            target_next_actions = target_qs.max(dim=3)[1].unsqueeze(-1).detach()
            next_q_total, q_attend_regs, _ = self.target_mix_net(target_qsa, next_s, target_next_actions)
        else:
            # 得到 target q，是inf出现的nan
            # target_qs[next_avail_a == 0.0] = float('-inf')
            # target_qs = target_qs.max(dim=3)[0]
            # 计算target Q_tot
            next_q_total = self.target_mix_net(target_qsa, next_s)

        target_q_total = r + self.args.gamma * next_q_total * (1 - done)
        # weights = torch.Tensor(np.ones(eval_q_total.shape))
        weights = torch.as_tensor(np.ones(eval_q_total.shape), dtype=torch.float, device=self.args.device)
        if self.wqmix > 0:
            # 1- 可以保证weights在 (0, 1]
            # TODO: 这里只说是 (0, 1] 之间，文中有介绍具体的参数设置
            # weights = torch.Tensor(1 - np.random.ranf(eval_q_total.shape))
            weights = torch.full(eval_q_total.shape, self.alpha, device=self.args.device)
            if self.args.alg == 'cwqmix':
                error = mask * (target_q_total - qstar_q_total)
            elif self.args.alg == 'owqmix':
                error = mask * (target_q_total - eval_q_total)
            else:
                raise Exception("模型不存在")
            weights[error > 0] = 1.
            # qstar 参数更新
            qstar_error = mask * (qstar_q_total - target_q_total.detach())

            qstar_loss = (qstar_error ** 2).sum() / mask.sum()
            # self.qstar_optimizer.zero_grad()
            # qstar_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.qstar_params, self.args.clip_norm)
            # self.qstar_optimizer.step()

        # 计算 TD error
        # TODO 这里权值detach有影响吗
        td_error = mask * (eval_q_total - target_q_total.detach())
        # if self.args.cuda:
        #     weights = weights.cuda()

        loss = (weights.detach() * td_error**2).sum() / mask.sum()
        if self.args.alg == 'qatten':
            loss += q_attend_regs
        elif self.wqmix > 0:
            loss += qstar_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.clip_norm)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_period == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            if self.wqmix > 0:
                self.qstar_target_rnn.load_state_dict(self.qstar_eval_rnn.state_dict())
                self.qstar_target_mix.load_state_dict(self.qstar_eval_mix.state_dict())

    def init_hidden(self, episode_num):
        """
        为每个episode中的每个agent都初始化一个eval_hidden，target_hidden
        :param episode_num:
        :return:
        """
        self.eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim),
                                       device=self.args.device)
        self.target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim),
                                         device=self.args.device)
        if self.wqmix > 0:
            self.qstar_eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim),
                                                 device=self.args.device)
            self.qstar_target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim),
                                                   device=self.args.device)

    def get_q(self, batch, episode_num, max_episode_len, wqmix=False):
        eval_qs, target_qs = [], []
        for trans_idx in range(max_episode_len):
            # 每个obs加上agent编号和last_action
            inputs, next_inputs = self.get_inputs(batch, episode_num, trans_idx)
            # 是否使用GPU
            # if self.args.cuda:
            #     # inputs = inputs.cuda()
            #     # next_inputs = next_inputs.cuda()
            #     if wqmix:
            #         self.qstar_eval_hidden = self.qstar_eval_hidden.cuda()
            #         self.qstar_target_hidden = self.qstar_target_hidden.cuda()
            #     else:
            #         self.eval_hidden = self.eval_hidden.cuda()
            #         self.target_hidden = self.target_hidden.cuda()
            # 得到q值
            if wqmix:
                eval_q, self.qstar_eval_hidden = self.qstar_eval_rnn(inputs, self.qstar_eval_hidden)
                target_q, self.qstar_target_hidden = self.qstar_target_rnn(next_inputs, self.qstar_target_hidden)
            else:
                eval_q, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
                target_q, self.target_hidden = self.target_rnn(next_inputs, self.target_hidden)
            # 形状变换
            eval_q = eval_q.view(episode_num, self.args.n_agents, -1)
            target_q = target_q.view(episode_num, self.args.n_agents, -1)
            # 添加这个transition 的信息
            eval_qs.append(eval_q)
            target_qs.append(target_q)
        # 将max_episode_len个(episode, n_agents, n_actions) 堆叠为 (episode, max_episode_len, n_agents, n_actions)
        eval_qs = torch.stack(eval_qs, dim=1)
        target_qs = torch.stack(target_qs, dim=1)
        return eval_qs, target_qs

    def get_inputs(self, batch, episode_num, trans_idx):
        # 取出所有episode上该trans_idx的经验，onehot_a要取出所有，因为要用到上一条
        obs, next_obs, onehot_a = batch['o'][:, trans_idx], \
                                  batch['next_o'][:, trans_idx], batch['onehot_a'][:]
        inputs, next_inputs = [], []
        inputs.append(obs)
        next_inputs.append(next_obs)
        # 给obs添加上一个动作，agent编号
        if self.args.last_action:
            if trans_idx == 0:
                inputs.append(torch.zeros_like(onehot_a[:, trans_idx], device=self.args.device))
            else:
                inputs.append(onehot_a[:, trans_idx-1])
            next_inputs.append(onehot_a[:, trans_idx])
        if self.args.reuse_network:
            """
            给数据增加agent编号，对于每个episode的数据，分为多个agent，每个agent编号为独热编码，
            这样对于所有agent的编号堆叠起来就是一个单位矩阵
            """
            inputs.append(torch.eye(self.args.n_agents, device=self.args.device).unsqueeze(0).expand(episode_num, -1, -1))
            next_inputs.append(torch.eye(self.args.n_agents, device=self.args.device).unsqueeze(0).expand(episode_num, -1, -1))
        # 将之前append的数据合并，得到形状为(episode_num*n_agents, obs*(n_actions*n_agents))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        next_inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in next_inputs], dim=1)
        return inputs, next_inputs

    def save_model(self, train_step):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        if type(train_step) == str:
            num = train_step
        else:
            num = str(train_step // self.args.save_model_period)

        torch.save(self.eval_mix_net.state_dict(), self.args.model_dir + '/' + num + '_'
                   + self.args.alg + '_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.args.model_dir + '/' + num + '_rnn_params.pkl')
