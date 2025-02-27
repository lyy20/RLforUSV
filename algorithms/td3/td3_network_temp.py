import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MultiheadAttention

# 定义一个包含LSTM和MHA的网络模块
class Network(nn.Module):
    def __init__(self, in_features, hidden_in, hidden_out, out_features, rnn_num_layers, rnn_hidden_size, device, actor=False, rnn=True):
        super(Network, self).__init__()
        self.rnn = rnn
        self.actor = actor
        self.device = device

        # LSTM层
        if self.rnn:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(rnn_hidden_size if self.rnn else in_features, hidden_in)
        self.fc2 = nn.Linear(hidden_in, hidden_out)
        self.fc3 = nn.Linear(hidden_out, out_features)

        # MHA层
        self.mha = MultiheadAttention(embed_dim=rnn_hidden_size if self.rnn else in_features, num_heads=4)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.rnn:
            # LSTM处理
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # 取最后一个时间步的输出

        # MHA处理
        x = x.unsqueeze(0)  # 添加一个维度以适应MHA的输入要求
        x, _ = self.mha(x, x, x)
        x = x.squeeze(0)

        # 全连接层处理
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        if self.actor:
            # 如果是actor网络，使用tanh激活函数
            x = torch.tanh(x)

        return x

# 修改SACAgent类
class SACAgent(nn.Module):
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic, lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device='cpu', rnn=True, alpha=0.2, automatic_entropy_tuning=True):
        super(SACAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device, actor=True, rnn=rnn).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn=rnn).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn=rnn).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)
        self.device = device

        # 初始化目标网络
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(out_actor).to(self.device)).item()
            self.log_alpha = (torch.zeros(1, requires_grad=True, device=self.device) + np.log(self.alpha)).detach().requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)

    def act(self, his, obs, noise=0.0):
        action = self.actor(his, obs)
        action = action + noise * torch.randn_like(action)
        return action

    def act_prob(self, his, obs):
        action = self.actor(his, obs)
        log_prob = torch.distributions.Normal(action, 0.1).log_prob(action)
        return action, log_prob

# 修改MASAC类
class MASAC:
    def __init__(self, num_agents=3, num_landmarks=1, num_obstacles=3, landmark_depth=15., discount_factor=0.95, tau=0.02, lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device='cpu', rnn=True, alpha=0.2, automatic_entropy_tuning=True, dim_1=64, dim_2=32):
        super(MASAC, self).__init__()

        in_actor = 1 * 2 * 2 + num_landmarks * 2 + (num_agents - 1) * 2 + num_landmarks + 1 * num_landmarks + 2 + 1 + 2 * num_obstacles + num_obstacles + num_obstacles
        hidden_in_actor = dim_2
        hidden_out_actor = int(hidden_in_actor / 2)
        out_actor = 1
        in_critic = in_actor * num_agents
        hidden_in_critic = dim_2
        hidden_out_critic = int(hidden_in_critic / 2)

        rnn_num_layers = 2
        rnn_hidden_size_actor = dim_1
        rnn_hidden_size_critic = dim_1

        self.masac_agent = [SACAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device, rnn=rnn, alpha=alpha, automatic_entropy_tuning=automatic_entropy_tuning) for _ in range(num_agents)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.iter_delay = 0
        self.policy_freq = 2
        self.num_agents = num_agents
        self.priority = 1.
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning

    def get_actors(self):
        actors = [sac_agent.actor for sac_agent in self.masac_agent]
        return actors

    def get_target_actors(self):
        target_actors = [sac_agent.target_actor for sac_agent in self.masac_agent]
        return target_actors

    def act(self, his_all_agents, obs_all_agents, noise=0.0):
        actions_next = [agent.act(his, obs, noise) for agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents)]
        return actions_next

    def act_prob(self, his_all_agents, obs_all_agents, noise=0.0):
        actions_next = []
        log_probs = []
        for sac_agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents):
            action, log_prob = sac_agent.act_prob(his, obs)
            log_prob = log_prob.view(-1)
            actions_next.append(action)
            log_probs.append(log_prob)
        return actions_next, log_probs

    def update(self, samples, agent_number, logger):
        # 代码省略，与原代码相同
        pass

    def update_targets(self):
        # 代码省略，与原代码相同
        pass