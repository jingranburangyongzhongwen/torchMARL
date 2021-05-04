# -*- coding: utf-8 -*-
"""
@Auth ： Kunfeng Li
@IDE ：PyCharm
"""
import torch
import os
from network.base_net import RNN
from network.qtran_mixer import *


class QTran:
    def __init__(self, args, itr):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        rnn_input_shape = self.obs_shape

        # 根据参数决定RNN的输入维度
        if args.last_action:
            rnn_input_shape += self.n_actions  # 当前agent的上一个动作的one_hot向量
        if args.reuse_network:
            rnn_input_shape += self.n_agents

        # 神经网络
        self.eval_rnn = RNN(rnn_input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(rnn_input_shape, args)

        # self.eval_joint_q = QtranQBase(args)  # Joint action-value network
        # self.target_joint_q = QtranQBase(args)
        # 默认值分解算法使用 qtran_base
        if self.args.alg == 'qtran_base':
            self.eval_joint_q = QtranQBase(args)
            self.target_joint_q = QtranQBase(args)
        elif self.args.alg == 'qtran_alt':
            # self.mixer = QTranAlt(args)
            # self.target_mixer = QTranAlt(args)
            raise Exception("Not supported yet.")
        else:
            raise Exception('QTRAN只有qtran_base！')

        self.v = QtranV(args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_joint_q.cuda()
            self.target_joint_q.cuda()
            self.v.cuda()

        # self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/' + str(itr)
        # self.model_dir = args.model_dir + '/' + args.map + '/' + args.alg + '_' + str(self.args.epsilon_anneal_steps // 10000) + 'w' + '/' + str(itr)
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.args.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.args.model_dir + '/rnn_net_params.pkl'
                path_joint_q = self.args.model_dir + '/joint_q_params.pkl'
                path_v = self.args.model_dir + '/v_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_joint_q.load_state_dict(torch.load(path_joint_q, map_location=map_location))
                self.v.load_state_dict(torch.load(path_v, map_location=map_location))
                print('Successfully load the model: {}, {} and {}'.format(path_rnn, path_joint_q, path_v))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())

        self.eval_parameters = list(self.eval_joint_q.parameters()) + \
                               list(self.v.parameters()) + \
                               list(self.eval_rnn.parameters())
        if args.optim == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QTRAN-base')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        # for key in batch.keys():  # 把batch里的数据转化成tensor
        #     if key == 'u':
        #         batch[key] = torch.tensor(batch[key], dtype=torch.long)
        #     else:
        #         batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        # 数据转为tensor
        for key in batch.keys():
            if key == 'a':
                batch[key] = torch.LongTensor(batch[key])
            else:
                batch[key] = torch.Tensor(batch[key])
        u, r, avail_u, avail_u_next, terminated = batch['a'], batch['r'],  batch['avail_a'], \
                                                  batch['next_avail_a'], batch['done']
        mask = (1 - batch["padded"].float()).squeeze(-1)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            avail_u = avail_u.cuda()
            avail_u_next = avail_u_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 得到每个agent对应的Q和hidden_states，维度为(episode个数, max_episode_len， n_agents， n_actions/hidden_dim)
        individual_q_evals, individual_q_targets, hidden_evals, hidden_targets = self._get_individual_q(batch, max_episode_len)

        # 得到当前时刻和下一时刻每个agent的局部最优动作及其one_hot表示
        individual_q_clone = individual_q_evals.clone()
        individual_q_clone[avail_u == 0.0] = - 999999
        individual_q_targets[avail_u_next == 0.0] = - 999999

        opt_onehot_eval = torch.zeros(*individual_q_clone.shape)
        opt_action_eval = individual_q_clone.argmax(dim=3, keepdim=True)
        opt_onehot_eval = opt_onehot_eval.scatter(-1, opt_action_eval[:, :].cpu(), 1)

        opt_onehot_target = torch.zeros(*individual_q_targets.shape)
        opt_action_target = individual_q_targets.argmax(dim=3, keepdim=True)
        opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:, :].cpu(), 1)

        # ---------------------------------------------L_td-------------------------------------------------------------
        # 计算joint_q和v
        # joint_q、v的维度为(episode个数, max_episode_len, 1), 而且joint_q在后面的l_nopt还要用到
        joint_q_evals, joint_q_targets, v = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_target)

        # loss
        y_dqn = r.squeeze(-1) + self.args.gamma * joint_q_targets * (1 - terminated.squeeze(-1))
        td_error = joint_q_evals - y_dqn.detach()
        l_td = ((td_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_td-------------------------------------------------------------

        # ---------------------------------------------L_opt------------------------------------------------------------
        # 将局部最优动作的Q值相加
        # 这里要使用individual_q_clone，它把不能执行的动作Q值改变了，使用individual_q_evals可能会使用不能执行的动作的Q值
        q_sum_opt = individual_q_clone.max(dim=-1)[0].sum(dim=-1)  # (episode个数, max_episode_len)

        # 重新得到joint_q_hat_opt，它和joint_q_evals的区别是前者输入的动作是局部最优动作，后者输入的动作是执行的动作
        # (episode个数, max_episode_len)
        joint_q_hat_opt, _, _ = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_eval, hat=True)
        opt_error = q_sum_opt - joint_q_hat_opt.detach() + v  # 计算l_opt时需要将joint_q_hat_opt固定
        l_opt = ((opt_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_opt------------------------------------------------------------

        # ---------------------------------------------L_nopt-----------------------------------------------------------
        # 每个agent的执行动作的Q值,(episode个数, max_episode_len, n_agents, 1)
        q_individual = torch.gather(individual_q_evals, dim=-1, index=u).squeeze(-1)
        q_sum_nopt = q_individual.sum(dim=-1)  # (episode个数, max_episode_len)

        nopt_error = q_sum_nopt - joint_q_evals.detach() + v  # 计算l_nopt时需要将joint_q_evals固定
        nopt_error = nopt_error.clamp(max=0)
        l_nopt = ((nopt_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_nopt-----------------------------------------------------------

        # print('l_td is {}, l_opt is {}, l_nopt is {}'.format(l_td, l_opt, l_nopt))
        loss = l_td + self.args.lambda_opt * l_opt + self.args.lambda_nopt * l_nopt
        # loss = l_td + self.args.lambda_opt * l_opt
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.clip_norm)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_period == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())

    def _get_individual_q(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets, hidden_evals, hidden_targets = [], [], [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_individual_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

            # 要用第一条经验把target网络的hidden_state初始化好，直接用第二条经验传入target网络不对
            if transition_idx == 0:
                _, self.target_hidden = self.target_rnn(inputs, self.eval_hidden)
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            hidden_eval, hidden_target = self.eval_hidden.clone(), self.target_hidden.clone()

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            hidden_eval = hidden_eval.view(episode_num, self.n_agents, -1)
            hidden_target = hidden_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            hidden_evals.append(hidden_eval)
            hidden_targets.append(hidden_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        hidden_evals = torch.stack(hidden_evals, dim=1)
        hidden_targets = torch.stack(hidden_targets, dim=1)
        return q_evals, q_targets, hidden_evals, hidden_targets

    def _get_individual_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['next_o'][:, transition_idx], batch['onehot_a'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_qtran(self, batch, hidden_evals, hidden_targets, local_opt_actions, hat=False):
        episode_num, max_episode_len, _, _ = hidden_targets.shape
        states = batch['s'][:, :max_episode_len]
        states_next = batch['next_s'][:, :max_episode_len]
        u_onehot = batch['onehot_a'][:, :max_episode_len]
        if self.args.cuda:
            states = states.cuda()
            states_next = states_next.cuda()
            u_onehot = u_onehot.cuda()
            hidden_evals = hidden_evals.cuda()
            hidden_targets = hidden_targets.cuda()
            local_opt_actions = local_opt_actions.cuda()
        if hat:
            # 神经网络输出的q_eval、q_target、v的维度为(episode_num * max_episode_len, 1)
            q_evals = self.eval_joint_q(states, hidden_evals, local_opt_actions)
            q_targets = None
            v = None

            # 把q_eval维度变回(episode_num, max_episode_len)
            q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
        else:
            q_evals = self.eval_joint_q(states, hidden_evals, u_onehot)
            q_targets = self.target_joint_q(states_next, hidden_targets, local_opt_actions)
            v = self.v(states, hidden_evals)
            # 把q_eval、q_target、v维度变回(episode_num, max_episode_len)
            q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
            q_targets = q_targets.view(episode_num, -1, 1).squeeze(-1)
            v = v.view(episode_num, -1, 1).squeeze(-1)

        return q_evals, q_targets, v

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        # num = str(train_step // self.args.save_cycle)
        # if not os.path.exists(self.model_dir):
        #     os.makedirs(self.model_dir)

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        if type(train_step) == str:
            num = train_step
        else:
            num = str(train_step // self.args.save_model_period)

        torch.save(self.eval_rnn.state_dict(),  self.args.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.eval_joint_q.state_dict(), self.args.model_dir + '/' + num + '_joint_q_params.pkl')
        torch.save(self.v.state_dict(), self.args.model_dir + '/' + num + '_v_params.pkl')

# class QTran:
#     def __init__(self, args, itr):
#         self.args = args
#         input_shape = self.args.obs_shape
#         # 调整RNN输入维度
#         if args.last_action:
#             input_shape += self.args.n_actions
#         if args.reuse_network:
#             input_shape += self.args.n_agents
#         # 设置网络
#         self.eval_rnn = RNN(input_shape, args)
#         self.target_rnn = RNN(input_shape, args)
#
#         # 默认值分解算法使用 qtran_base
#         if self.args.alg == 'qtran_base':
#             self.mixer = QTranBase(args)
#             self.target_mixer = QTranBase(args)
#         elif self.args.alg == 'qtran_alt':
#             # self.mixer = QTranAlt(args)
#             # self.target_mixer = QTranAlt(args)
#             raise Exception("Not supported yet.")
#
#         # 是否使用GPU
#         if args.cuda:
#             self.eval_rnn.cuda()
#             self.target_rnn.cuda()
#             self.mixer.cuda()
#             self.target_mixer.cuda()
#         # 是否加载模型
#         self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/' + str(itr)
#         if args.load_model:
#             if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
#                 path_rnn = self.model_dir + '/rnn_net_params.pkl'
#                 path_mix = self.model_dir + '/' + self.args.alg + '_net_params.pkl'
#
#                 map_location = 'cuda:0' if args.cuda else 'cpu'
#
#                 self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
#                 self.mixer.load_state_dict(torch.load(path_mix, map_location=map_location))
#                 print('成功加载模型 %s' % path_rnn + ' 和 %s' % path_mix)
#             else:
#                 raise Exception("模型不存在")
#         # 令target网络与eval网络参数相同
#         self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
#         self.target_mixer.load_state_dict(self.mixer.state_dict())
#         # 获取所有参数
#         self.eval_params = list(self.eval_rnn.parameters()) + list(self.mixer.parameters())
#         # 学习过程中要为每个episode的每个agent维护一个eval_hidden，执行过程中要为每个agetn维护一个eval_hidden
#         self.eval_hidden = None
#         self.target_hidden = None
#         self.hidden_states = None
#         self.target_hidden_states = None
#
#         # 监控
#         # torch.autograd.set_detect_anomaly(True)
#
#         # 获取优化器
#         if args.optim == 'RMS':
#             self.optimizer = torch.optim.RMSprop(self.eval_params, lr=args.lr)
#         else:
#             self.optimizer = torch.optim.Adam(self.eval_params)
#         print("值分解算法 " + self.args.alg + " 初始化")
#
#     def learn(self, batch, max_episode_len, train_step):
#         """
#         在learn的时候，抽取到的数据是四维的，四个维度分别为
#         1——第几个episode
#         2——episode中第几个transition
#         3——第几个agent的数据
#         4——具体obs维度。
#         因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
#         hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，
#         然后一次给神经网络传入每个episode的同一个位置的transition
#         :param batch:
#         :param max_episode_len:
#         :param train_step:
#         :param epsilon:
#         :return:
#         """
#         # 获得episode的数目
#         episode_num = batch['o'].shape[0]
#         # 初始化隐藏状态
#         self.init_hidden(episode_num)
#         # 数据转为tensor
#         for key in batch.keys():
#             if key == 'a':
#                 batch[key] = torch.LongTensor(batch[key])
#             else:
#                 batch[key] = torch.Tensor(batch[key])
#         s, next_s, a, r, avail_a, next_avail_a, done = batch['s'], batch['next_s'], batch['a'], \
#                                                        batch['r'], batch['avail_a'], batch['next_avail_a'], \
#                                                        batch['done']
#         # 避免填充的产生 TD-error 影响训练
#         mask = 1 - batch["padded"].float()
#         # 获取当前与下个状态的q值，（episode, max_episode_len, n_agents, n_actions）
#         eval_qs, target_qs = self.get_q(batch, episode_num, max_episode_len)
#         # 是否使用GPU
#         if self.args.cuda:
#             a = a.cuda()
#             r = r.cuda()
#             done = done.cuda()
#             mask = mask.cuda()
#             if 'qmix' in self.args.alg:
#                 s = s.cuda()
#                 next_s = next_s.cuda()
#         # 得到每个动作对应的 q 值
#         eval_qs[avail_a == 0.0] = -9999999
#         eval_qsa = torch.gather(eval_qs, dim=3, index=a).squeeze(3)
#         max_actions_qvals, max_actions_current = eval_qs[:, :].max(dim=3, keepdim=True)
#
#         target_qs[next_avail_a == 0.0] = -9999999
#         # target_qsa = target_qs.max(dim=3)[0]
#         target_max_actions = target_qs.argmax(dim=3).unsqueeze(3)
#         td_loss, opt_loss, nopt_loss = 0, 0, 0
#         # TODO 有待完善 Qtran
#         if self.args.alg == "qtran_base":
#             # -- TD Loss --
#             # Joint-action Q-Value estimates
#             joint_qs, vs = self.mixer(episode_num, max_episode_len, batch, self.hidden_states)
#
#             # Need to argmax across the target agents' actions to compute target joint-action Q-Values
#             # if self.args.double_q:
#             #     max_actions_current_ = torch.zeros(
#             #         size=(episode_num, max_episode_len, self.args.n_agents, self.args.n_actions),
#             #         device=batch.device)
#             #     max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
#             #     max_actions_onehot = max_actions_current_onehot
#             # else:
#             max_actions = torch.zeros(
#                 size=(episode_num, max_episode_len, self.args.n_agents, self.args.n_actions),
#             )
#             if self.args.cuda:
#                 max_actions.cuda()
#             max_actions_onehot = max_actions.scatter(3, target_max_actions, 1)
#             target_joint_qs, target_vs = self.target_mixer(episode_num, max_episode_len, batch,
#                                                            hidden_states=self.target_hidden_states,
#                                                            actions=max_actions_onehot)
#
#             # Td loss targets
#             td_targets = r.reshape(-1, 1) + self.args.gamma * (1 - done.reshape(-1, 1)) * target_joint_qs
#             td_error = (joint_qs - td_targets.detach())
#             masked_td_error = td_error * mask.reshape(-1, 1)
#             td_loss = (masked_td_error ** 2).sum() / mask.sum()
#             # -- TD Loss --
#
#             # -- Opt Loss --
#             # Argmax across the current agents' actions
#             # if not self.args.double_q:  # Already computed if we're doing double Q-Learning
#             max_actions_current_ = torch.zeros(
#                 size=(episode_num, max_episode_len, self.args.n_agents, self.args.n_actions))
#             if self.args.cuda:
#                 max_actions_current_.cuda()
#             max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current, 1)
#             # Don't use the target network and target agent max actions as per author's email
#             max_joint_qs, _ = self.mixer(episode_num, max_episode_len, batch, self.hidden_states,
#                                          actions=max_actions_current_onehot)
#
#             # max_actions_qvals = torch.gather(mac_out[:, :-1], dim=3, index=max_actions_current[:,:-1])
#             opt_error = max_actions_qvals.sum(dim=2).reshape(-1, 1) - max_joint_qs.detach() + vs
#             masked_opt_error = opt_error * mask.reshape(-1, 1)
#             opt_loss = (masked_opt_error ** 2).sum() / mask.sum()
#             # -- Opt Loss --
#
#             # -- Nopt Loss --
#             # target_joint_qs, _ = self.target_mixer(batch[:, :-1])
#             nopt_values = eval_qsa.sum(dim=2).reshape(-1,
#                                                       1) - joint_qs.detach() + vs  # Don't use target networks here either
#             nopt_error = nopt_values.clamp(max=0)
#             masked_nopt_error = nopt_error * mask.reshape(-1, 1)
#             nopt_loss = (masked_nopt_error ** 2).sum() / mask.sum()
#             # -- Nopt loss --
#
#         elif self.args.alg == "qtran_alt":
#             raise Exception("Not supported yet.")
#
#         loss = td_loss + self.args.opt_loss * opt_loss + self.args.nopt_min_loss * nopt_loss
#
#         # # 需要先把不行动作的mask掉
#         # target_qs[next_avail_a == 0.0] = -9999999
#         # target_q_total = r + self.args.gamma * next_q_total * (1 - done)
#         # weights = torch.Tensor(np.ones(eval_q_total.shape))
#         #
#         # # 计算 TD error
#         # # TODO 这里权值detach有影响吗
#         # td_error = mask * (eval_q_total - target_q_total.detach())
#         # if self.args.cuda:
#         #     weights = weights.cuda()
#         # loss = (weights.detach() * td_error**2).sum() / mask.sum()
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.clip_norm)
#         self.optimizer.step()
#
#         if train_step > 0 and train_step % self.args.target_update_period == 0:
#             self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
#             self.target_mixer.load_state_dict(self.mixer.state_dict())
#
#     def init_hidden(self, episode_num):
#         """
#         为每个episode中的每个agent都初始化一个eval_hidden，target_hidden
#         :param episode_num:
#         :return:
#         """
#         self.eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))
#         self.target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))
#
#     def get_q(self, batch, episode_num, max_episode_len):
#         eval_qs, target_qs = [], []
#         self.hidden_states = torch.zeros((max_episode_len, episode_num,
#                                           self.args.n_agents, self.args.rnn_hidden_dim))
#         self.target_hidden_states = torch.zeros((max_episode_len, episode_num,
#                                                  self.args.n_agents, self.args.rnn_hidden_dim))
#         for trans_idx in range(max_episode_len):
#             self.hidden_states[trans_idx] = self.eval_hidden.reshape(episode_num,
#                                                                      self.args.n_agents, self.args.rnn_hidden_dim)
#             self.target_hidden_states[trans_idx] = self.target_hidden.reshape(episode_num,
#                                                                      self.args.n_agents, self.args.rnn_hidden_dim)
#             # 每个obs加上agent编号和last_action
#             inputs, next_inputs = self.get_inputs(batch, episode_num, trans_idx)
#             # 是否使用GPU
#             if self.args.cuda:
#                 inputs = inputs.cuda()
#                 next_inputs = next_inputs.cuda()
#                 self.eval_hidden = self.eval_hidden.cuda()
#                 self.target_hidden = self.target_hidden.cuda()
#             # 得到q值
#             eval_q, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
#             target_q, self.target_hidden = self.target_rnn(next_inputs, self.target_hidden)
#             # 形状变换
#             eval_q = eval_q.view(episode_num, self.args.n_agents, -1)
#             target_q = target_q.view(episode_num, self.args.n_agents, -1)
#             # 添加这个transition 的信息
#             eval_qs.append(eval_q)
#             target_qs.append(target_q)
#         # 将max_episode_len个(episode, n_agents, n_actions) 堆叠为 (episode, max_episode_len, n_agents, n_actions)
#         eval_qs = torch.stack(eval_qs, dim=1)
#         target_qs = torch.stack(target_qs, dim=1)
#
#         self.hidden_states = self.hidden_states.permute(1, 0, 2, 3)
#         self.target_hidden_states = self.target_hidden_states.permute(1, 0, 2, 3)
#         return eval_qs, target_qs
#
#     def get_inputs(self, batch, episode_num, trans_idx):
#         # 取出所有episode上该trans_idx的经验，onehot_a要取出所有，因为要用到上一条
#         obs, next_obs, onehot_a = batch['o'][:, trans_idx], \
#                                   batch['next_o'][:, trans_idx], batch['onehot_a'][:]
#         inputs, next_inputs = [], []
#         inputs.append(obs)
#         next_inputs.append(next_obs)
#         # 给obs添加上一个动作，agent编号
#         if self.args.last_action:
#             if trans_idx == 0:
#                 inputs.append(torch.zeros_like(onehot_a[:, trans_idx]))
#             else:
#                 inputs.append(onehot_a[:, trans_idx - 1])
#             next_inputs.append(onehot_a[:, trans_idx])
#         if self.args.reuse_network:
#             """
#             给数据增加agent编号，对于每个episode的数据，分为多个agent，每个agent编号为独热编码，
#             这样对于所有agent的编号堆叠起来就是一个单位矩阵
#             """
#             inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
#             next_inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
#         # 将之前append的数据合并，得到形状为(episode_num*n_agents, obs*(n_actions*n_agents))
#         inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
#         next_inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in next_inputs], dim=1)
#         return inputs, next_inputs
#
#     def save_model(self, train_step):
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
#
#         if type(train_step) == str:
#             num = train_step
#         else:
#             num = str(train_step // self.args.save_model_period)
#
#         torch.save(self.mixer.state_dict(), self.model_dir + '/' + num + '_'
#                    + self.args.alg + '_net_params.pkl')
#         torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_params.pkl')
