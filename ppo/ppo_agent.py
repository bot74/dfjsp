from torch import nn
import torch
from torch.distributions import Normal #引入torch里的正态分布

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"computing device: {device}")

#Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor).__init__()
        self.fc1 = nn.Linear(state_dim. hidden_dim) #全连接1
        self.fc2 == nn.Linear(hidden_dim, hidden_dim) #全连接2
        #测试环境是连续的动作环境，用分步采样的方式来得到动作，假设正态分布，所以构建了fc_mean,fc_std
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU() #激活函数
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        #demo环境摇摆力度（-2，2）

    def forward(self, x): #前向传播
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2 #tan范围是-1,1 ，因此乘上2
        std = self.softplus(self.fc_std(x)) + 1e-3 #防止因为std太接近0产生奇怪结果，+1e-3修正一下

        return mean, std

    def select_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
            normal_dist = Normal(mu, sigma) #确定正态分布？
            action = normal_dist.sample() #从正态分布中抽样来取出action
            action = action.clamp(-2.0, 2.0) #限制action值以防超出场景范围

        return action #返回选择的动作
#Critic网络
class Critic:

#经验获取？
class ReplayMemory:

#Actor-Critic网络会被统一封装在PPOAgent类中
class PPOAgent: