import torch
import os
from network.base_net import RNN
from network.dmaq_general import DMAQer
from network.dmaq_qatten import DMAQ_QattenMixer
import numpy as np


class DMAQ_qattenLearner:
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

        # 默认值分解算法使用 dmaq
        if self.args.alg == 'dmaq_qatten':
            self.mixer = DMAQ_QattenMixer()
            self.target_mixer = DMAQ_QattenMixer()
        elif self.args.alg == 'dmaq':
            self.mixer = DMAQer(args)
            self.target_mixer = DMAQer(args)
        else:
            raise Exception('Unsupported!')

        # 是否使用GPU
        if args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.mixer.cuda()
            self.target_mixer.cuda()
        # 是否加载模型
        # self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/' + str(itr)
        # self.model_dir = args.model_dir + '/' + args.map + '/' + args.alg + '_' + str(self.args.epsilon_anneal_steps // 10000) + 'w' + '/' + str(itr)
        if args.load_model:
            if os.path.exists(self.args.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.args.model_dir + '/rnn_net_params.pkl'
                path_mix = self.args.model_dir + '/' + self.args.alg + '_net_params.pkl'

                map_location = 'cuda:0' if args.cuda else 'cpu'

                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.mixer.load_state_dict(torch.load(path_mix, map_location=map_location))

                print('成功加载模型 %s' % path_rnn + ' 和 %s' % path_mix)
            else:
                raise Exception("模型不存在")
        # 令target网络与eval网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        # 获取所有参数
        self.eval_params = list(self.eval_rnn.parameters()) + list(self.mixer.parameters())
        # 学习过程中要为每个episode的每个agent维护一个eval_hidden，执行过程中要为每个agetn维护一个eval_hidden
        self.eval_hidden = None
        self.target_hidden = None

        # 获取优化器
        if args.optim == 'RMS':
            self.optimizer = torch.optim.RMSprop(self.eval_params, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_params)
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
        for key in batch.keys():
            if key == 'a':
                batch[key] = torch.LongTensor(batch[key])
            else:
                batch[key] = torch.Tensor(batch[key])
        s, next_s, a, r, avail_a, next_avail_a, done, actions_onehot = batch['s'], batch['next_s'], batch['a'], \
                                                                       batch['r'], batch['avail_a'], batch[
                                                                           'next_avail_a'], \
                                                                       batch['done'], batch['onehot_a']
        # 避免填充的产生 TD-error 影响训练
        mask = 1 - batch["padded"].float()
        # 获取当前与下个状态的q值，（episode, max_episode_len, n_agents, n_actions）
        eval_qs, target_qs = self.get_q(batch, episode_num, max_episode_len)
        # 是否使用GPU
        if self.args.cuda:
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            mask = mask.cuda()
            s = s.cuda()
            next_s = next_s.cuda()
            actions_onehot = actions_onehot.cuda()
        # 得到每个动作对应的 q 值
        eval_qsa = torch.gather(eval_qs, dim=3, index=a).squeeze(3)
        max_action_qvals = eval_qs.max(dim=3)[0]

        target_qs[next_avail_a == 0.0] = -9999999
        target_qsa = target_qs.max(dim=3)[0]
        # target_max_actions = target_qs.argmax(dim=3).unsqueeze(3)
        # 计算Q_tot
        q_attend_regs = None
        if self.args.alg == "dmaq_qatten":
            ans_chosen, q_attend_regs, head_entropies = self.mixer(eval_qsa, s, is_v=True)
            ans_adv, _, _ = self.mixer(eval_qsa, s, actions=actions_onehot,
                                       max_q_i=max_action_qvals, is_v=False)
            eval_qsa = ans_chosen + ans_adv
        else:
            ans_chosen = self.mixer(eval_qsa, s, is_v=True)
            ans_adv = self.mixer(eval_qsa, s, actions=actions_onehot,
                                 max_q_i=max_action_qvals, is_v=False)
            eval_qsa = ans_chosen + ans_adv

        # if self.args.double_q:
        #     if self.args.self.mixer == "dmaq_qatten":
        #         target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
        #         target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
        #                                              actions=cur_max_actions_onehot,
        #                                              max_q_i=target_max_qvals, is_v=False)
        #         target_max_qvals = target_chosen + target_adv
        #     else:
        #     target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
        #     target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
        #                                    actions=cur_max_actions_onehot,
        #                                    max_q_i=target_max_qvals, is_v=False)
        #     target_max_qvals = target_chosen + target_adv
        # else:
        target_max_qvals = self.target_mixer(target_qsa, next_s, is_v=True)

        # Calculate 1-step Q-Learning targets
        targets = r + self.args.gamma * (1 - done) * target_max_qvals

        # Td-error
        td_error = (eval_qsa - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.args.alg == "dmaq_qatten":
            loss += q_attend_regs
        # else:
        #     loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.clip_norm)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_period == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def init_hidden(self, episode_num):
        """
        为每个episode中的每个agent都初始化一个eval_hidden，target_hidden
        :param episode_num:
        :return:
        """
        self.eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))

    def get_q(self, batch, episode_num, max_episode_len, ):
        eval_qs, target_qs = [], []
        for trans_idx in range(max_episode_len):
            # 每个obs加上agent编号和last_action
            inputs, next_inputs = self.get_inputs(batch, episode_num, trans_idx)
            # 是否使用GPU
            if self.args.cuda:
                inputs = inputs.cuda()
                next_inputs = next_inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            # 得到q值
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
                inputs.append(torch.zeros_like(onehot_a[:, trans_idx]))
            else:
                inputs.append(onehot_a[:, trans_idx - 1])
            next_inputs.append(onehot_a[:, trans_idx])
        if self.args.reuse_network:
            """
            给数据增加agent编号，对于每个episode的数据，分为多个agent，每个agent编号为独热编码，
            这样对于所有agent的编号堆叠起来就是一个单位矩阵
            """
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            next_inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
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

        torch.save(self.mixer.state_dict(), self.args.model_dir + '/' + num + '_'
                   + self.args.alg + '_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.args.model_dir + '/' + num + '_rnn_params.pkl')

# class DMAQ_qattenLearner:
#     def __init__(self, mac, scheme, logger, args):
#         self.args = args
#         self.mac = mac
#         self.logger = logger
# 
#         self.params = list(mac.parameters())
# 
#         self.last_target_update_episode = 0
# 
#         self.mixer = None
#         if args.mixer is not None:
#             if args.mixer == "dmaq":
#                 self.mixer = DMAQer(args)
#             elif args.mixer == 'dmaq_qatten':
#                 self.mixer = DMAQ_QattenMixer(args)
#             else:
#                 raise ValueError("Mixer {} not recognised.".format(args.mixer))
#             self.params += list(self.mixer.parameters())
#             self.target_mixer = copy.deepcopy(self.mixer)
# 
#         self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
# 
#         # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
#         self.target_mac = copy.deepcopy(mac)
# 
#         self.log_stats_t = -self.args.learner_log_interval - 1
# 
#         self.n_actions = self.args.n_actions
# 
#     def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
#                   show_demo=False, save_data=None):
#         # Get the relevant quantities
#         rewards = batch["reward"][:, :-1]
#         actions = batch["actions"][:, :-1]
#         terminated = batch["terminated"][:, :-1].float()
#         mask = batch["filled"][:, :-1].float()
#         mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
#         avail_actions = batch["avail_actions"]
#         actions_onehot = batch["actions_onehot"][:, :-1]
# 
#         # Calculate estimated Q-Values
#         mac_out = []
#         mac.init_hidden(batch.batch_size)
#         for t in range(batch.max_seq_length):
#             agent_outs = mac.forward(batch, t=t)
#             mac_out.append(agent_outs)
#         mac_out = th.stack(mac_out, dim=1)  # Concat over time
# 
#         # Pick the Q-Values for the actions taken by each agent
#         eval_qsa = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
# 
#         x_mac_out = mac_out.clone().detach()
#         x_mac_out[avail_actions == 0] = -9999999
#         max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
# 
#         max_action_index = max_action_index.detach().unsqueeze(3)
#         is_max_action = (max_action_index == actions).int().float()
# 
#         # Calculate the Q-Values necessary for the target
#         target_mac_out = []
#         self.target_mac.init_hidden(batch.batch_size)
#         for t in range(batch.max_seq_length):
#             target_agent_outs = self.target_mac.forward(batch, t=t)
#             target_mac_out.append(target_agent_outs)
# 
#         # We don't need the first timesteps Q-Value estimate for calculating targets
#         target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
# 
#         # Mask out unavailable actions
#         target_mac_out[avail_actions[:, 1:] == 0] = -9999999
# 
#         # Max over target Q-Values
#         if self.args.double_q:
#             # Get actions that maximise live Q (for double q-learning)
#             mac_out_detach = mac_out.clone().detach()
#             mac_out_detach[avail_actions == 0] = -9999999
#             cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
#             target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
#             target_max_qvals = target_mac_out.max(dim=3)[0]
#             target_next_actions = cur_max_actions.detach()
# 
#             cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
#             cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
#         else:
#             # Calculate the Q-Values necessary for the target
#             target_mac_out = []
#             self.target_mac.init_hidden(batch.batch_size)
#             for t in range(batch.max_seq_length):
#                 target_agent_outs = self.target_mac.forward(batch, t=t)
#                 target_mac_out.append(target_agent_outs)
#             # We don't need the first timesteps Q-Value estimate for calculating targets
#             target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
#             target_max_qvals = target_mac_out.max(dim=3)[0]
# 
#         # Mix
#         if mixer is not None:
#             if self.args.mixer == "dmaq_qatten":
#                 ans_chosen, q_attend_regs, head_entropies = \
#                     mixer(eval_qsa, batch["state"][:, :-1], is_v=True)
#                 ans_adv, _, _ = mixer(eval_qsa, batch["state"][:, :-1], actions=actions_onehot,
#                                       max_q_i=max_action_qvals, is_v=False)
#                 eval_qsa = ans_chosen + ans_adv
#             else:
#                 ans_chosen = mixer(eval_qsa, batch["state"][:, :-1], is_v=True)
#                 ans_adv = mixer(eval_qsa, batch["state"][:, :-1], actions=actions_onehot,
#                                 max_q_i=max_action_qvals, is_v=False)
#                 eval_qsa = ans_chosen + ans_adv
# 
#             if self.args.double_q:
#                 if self.args.mixer == "dmaq_qatten":
#                     target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
#                     target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
#                                                          actions=cur_max_actions_onehot,
#                                                          max_q_i=target_max_qvals, is_v=False)
#                     target_max_qvals = target_chosen + target_adv
#                 else:
#                     target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
#                     target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
#                                                    actions=cur_max_actions_onehot,
#                                                    max_q_i=target_max_qvals, is_v=False)
#                     target_max_qvals = target_chosen + target_adv
#             else:
#                 target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)
# 
#         # Calculate 1-step Q-Learning targets
#         targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
# 
#         # Td-error
#         td_error = (eval_qsa - targets.detach())
# 
#         mask = mask.expand_as(td_error)
# 
#         # 0-out the targets that came from padded data
#         masked_td_error = td_error * mask
# 
#         # Normal L2 loss, take mean over actual data
#         if self.args.mixer == "dmaq_qatten":
#             loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
#         else:
#             loss = (masked_td_error ** 2).sum() / mask.sum()
# 
#         masked_hit_prob = th.mean(is_max_action, dim=2) * mask
#         hit_prob = masked_hit_prob.sum() / mask.sum()
# 
#         # Optimise
#         optimiser.zero_grad()
#         loss.backward()
#         grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
#         optimiser.step()
# 
#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             self.logger.log_stat("loss", loss.item(), t_env)
#             self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
#             self.logger.log_stat("grad_norm", grad_norm, t_env)
#             mask_elems = mask.sum().item()
#             self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
#             self.logger.log_stat("q_taken_mean",
#                                  (eval_qsa * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
#             self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
#                                  t_env)
#             self.log_stats_t = t_env
# 
#     def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
#         self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
#                        show_demo=show_demo, save_data=save_data)
#         if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
#             self._update_targets()
#             self.last_target_update_episode = episode_num
# 
#     def _update_targets(self):
#         self.target_mac.load_state(self.mac)
#         if self.mixer is not None:
#             self.target_mixer.load_state_dict(self.mixer.state_dict())
#         self.logger.console_logger.info("Updated target network")
# 
#     def cuda(self):
#         self.mac.cuda()
#         self.target_mac.cuda()
#         if self.mixer is not None:
#             self.mixer.cuda()
#             self.target_mixer.cuda()
# 
#     def save_models(self, path):
#         self.mac.save_models(path)
#         if self.mixer is not None:
#             th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
#         th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
# 
#     def load_models(self, path):
#         self.mac.load_models(path)
#         # Not quite right but I don't want to save target networks
#         self.target_mac.load_models(path)
#         if self.mixer is not None:
#             self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
#             self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
#                                                       map_location=lambda storage, loc: storage))
#         self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
