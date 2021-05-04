import torch
import torch.nn as nn
import numpy as np


# counterfactual joint networks, 输入state、所有agent的hidden_state、其他agent的动作、自己的编号，输出自己所有动作对应的联合Q值
class QtranQAlt(nn.Module):
    def __init__(self, args):
        super(QtranQAlt, self).__init__()
        self.args = args

        # 对每个agent的action进行编码
        self.action_encoding = nn.Sequential(nn.Linear(self.args.n_actions, self.args.n_actions),
                                             nn.ReLU(),
                                             nn.Linear(self.args.n_actions, self.args.n_actions))

        # 对每个agent的hidden_state进行编码
        self.hidden_encoding = nn.Sequential(nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim))

        # 编码求和之后输入state、所有agent的hidden_state之和、其他agent的action之和, state包括当前agent的编号
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim + self.args.n_agents
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.n_actions))

    # 因为所有时刻所有agent的hidden_states在之前已经计算好了，所以联合Q值可以一次计算所有transition的，不需要一条一条计算。
    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        # state的shape为(episode_num, max_episode_len, n_agents, state_shape+n_agents)，包括了当前agent的编号
        episode_num, max_episode_len, n_agents, n_actions = actions.shape

        # 对每个agent的action进行编码
        action_encoding = self.action_encoding(actions.reshape(-1, n_actions))
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_actions)

        # 对每个agent的hidden_state进行编码
        hidden_encoding = self.hidden_encoding(hidden_states.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num, max_episode_len, n_agents, self.args.rnn_hidden_dim)

        # 所有agent的hidden_encoding相加
        hidden_encoding = hidden_encoding.sum(dim=-2)  # (episode_num, max_episode_len, rnn_hidden_dim)
        hidden_encoding = hidden_encoding.unsqueeze(-2).expand(-1, -1, n_agents, -1)  # (episode_num, max_episode_len, n_agents， rnn_hidden_dim)

        # 对于每个agent，其他agent的action_encoding相加
        # 先让最后一维包含所有agent的动作
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, 1, n_agents * n_actions)
        action_encoding = action_encoding.repeat(1, 1, n_agents, 1)  # 此时每个agent都有了所有agent的动作
        # 把每个agent自己的动作置0
        action_mask = (1 - torch.eye(n_agents))  # torch.eye（）生成一个二维对角矩阵
        action_mask = action_mask.view(-1, 1).repeat(1, n_actions).view(n_agents, -1)
        if self.args.cuda:
            action_mask = action_mask.cuda()
        action_encoding = action_encoding * action_mask.unsqueeze(0).unsqueeze(0)
        # 因为现在所有agent的动作都在最后一维，不能直接加。所以先扩展一维，相加后再去掉
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_agents, n_actions)
        action_encoding = action_encoding.sum(dim=-2)  # (episode_num, max_episode_len, n_agents， rnn_hidden_dim)

        inputs = torch.cat([state, hidden_encoding, action_encoding], dim=-1)
        q = self.q(inputs)
        return q


# Joint action-value network， 输入state,所有agent的hidden_state，所有agent的动作，输出对应的联合Q值
class QtranQBase(nn.Module):
    def __init__(self, args):
        super(QtranQBase, self).__init__()
        self.args = args
        # action_encoding对输入的每个agent的hidden_state和动作进行编码，从而将所有agents的hidden_state和动作相加得到近似的联合hidden_state和动作
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                             nn.ReLU(),
                                             nn.Linear(ae_input, ae_input))

        # 编码求和之后输入state、所有agent的hidden_state和动作之和
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    # 因为所有时刻所有agent的hidden_states在之前已经计算好了，所以联合Q值可以一次计算所有transition的，不需要一条一条计算。
    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)  # 变回n_agents维度用于求和
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)

        inputs = torch.cat([state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1)
        q = self.q(inputs)
        return q


# 输入当前的state与所有agent的hidden_state, 输出V值
class QtranV(nn.Module):
    def __init__(self, args):
        super(QtranV, self).__init__()
        self.args = args

        # hidden_encoding对输入的每个agent的hidden_state编码，从而将所有agents的hidden_state相加得到近似的联合hidden_state
        hidden_input = self.args.rnn_hidden_dim
        self.hidden_encoding = nn.Sequential(nn.Linear(hidden_input, hidden_input),
                                             nn.ReLU(),
                                             nn.Linear(hidden_input, hidden_input))

        # 编码求和之后输入state、所有agent的hidden_state之和
        v_input = self.args.state_shape + self.args.rnn_hidden_dim
        self.v = nn.Sequential(nn.Linear(v_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    def forward(self, state, hidden):
        episode_num, max_episode_len, n_agents, _ = hidden.shape
        state = state.reshape(episode_num * max_episode_len, -1)
        hidden_encoding = self.hidden_encoding(hidden.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num * max_episode_len, n_agents, -1).sum(dim=-2)
        inputs = torch.cat([state, hidden_encoding], dim=-1)
        v = self.v(inputs)
        return v


class QTranBase(nn.Module):
    def __init__(self, args):
        super(QTranBase, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.arch = self.args.qtran_arch  # QTran architecture

        self.embed_dim = args.mixing_embed_dim

        # Q(s,u)
        if self.arch == "coma_critic":
            # Q takes [state, u] as input
            q_input_size = self.state_dim + (self.n_agents * self.n_actions)
        elif self.arch == "qtran_paper":
            # Q takes [state, agent_action_observation_encodings]
            q_input_size = self.state_dim + self.args.rnn_hidden_dim + self.n_actions
        else:
            raise Exception("{} is not a valid QTran architecture".format(self.arch))

        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            ae_input = self.args.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                 nn.ReLU(),
                                                 nn.Linear(ae_input, ae_input))
        elif self.args.network_size == "big":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            ae_input = self.args.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                 nn.ReLU(),
                                                 nn.Linear(ae_input, ae_input))
        else:
            assert False

    def forward(self, bs, ts, batch, hidden_states, actions=None):
        # bs = batch.batch_size
        # ts = batch.max_seq_length

        states = batch["s"].reshape(bs * ts, self.state_dim)

        if self.arch == "coma_critic":
            if actions is None:
                # Use the actions taken by the agents
                actions = batch["onehot_a"].reshape(bs * ts, self.n_agents * self.n_actions)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents * self.n_actions)
            inputs = torch.cat([states, actions], dim=1)
        elif self.arch == "qtran_paper":
            if actions is None:
                # Use the actions taken by the agents
                actions = batch["onehot_a"].reshape(bs * ts, self.n_agents, self.n_actions)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents, self.n_actions)

            hidden_states = hidden_states.reshape(bs * ts, self.n_agents, -1)
            agent_state_action_input = torch.cat([hidden_states, actions], dim=2)
            agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(bs * ts * self.n_agents, -1)).reshape(bs * ts, self.n_agents, -1)
            agent_state_action_encoding = agent_state_action_encoding.sum(dim=1)  # Sum across agents

            inputs = torch.cat([states, agent_state_action_encoding], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["s"].reshape(bs * ts, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs


class QTranAlt(nn.Module):
    def __init__(self, args):
        super(QTranAlt, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        # Q(s,-,u-i)
        # Q takes [state, u-i, i] as input
        q_input_size = self.state_dim + (self.n_agents * self.n_actions) + self.n_agents

        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.n_actions))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        elif self.args.network_size == "big":
             # Adding another layer
             self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.n_actions))
            # V(s)
             self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        else:
            assert False

    def forward(self, bs, ts, batch, masked_actions=None):
        # bs = batch.batch_size
        # ts = batch.max_seq_length
        # Repeat each state n_agents times
        repeated_states = batch["state"].repeat(1, 1, self.n_agents).view(-1, self.state_dim)

        if masked_actions is None:
            actions = batch["onehot_a"].repeat(1, 1, self.n_agents, 1)
            agent_mask = (1 - torch.eye(self.n_agents, device=batch.device))
            agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions)#.view(self.n_agents, -1)
            masked_actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)
            masked_actions = masked_actions.view(-1, self.n_agents * self.n_actions)

        agent_ids = torch.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).repeat(bs, ts, 1, 1).view(-1, self.n_agents)

        inputs = torch.cat([repeated_states, masked_actions, agent_ids], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["s"].repeat(1,1,self.n_agents).view(-1, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs


