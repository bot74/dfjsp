from torch import nn
import torch
from torch.distributions import Normal #引入torch里的正态分布
import numpy as np
import torch.optim as optim #引入torch自带的优化器

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"computing device: {device}")

#Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        #在PyTorch的神经网络类中，super()的作用是 调用父类（nn.Module）的构造函数，
        # 确保父类的初始化逻辑被正确执行。这是神经网络模块能正常工作的基础。
        # super()返回父类(nn.Module)的实例，使你能够调用父类的方法。
        # __init__()显式调用父类的构造函数
        super().__init__() # deepseek: ✅ 等效于 super(Critic, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(state_dim, hidden_dim) #全连接1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) #全连接2
        
        # 测试环境是连续的动作环境，用分步采样的方式来得到动作，假设正态分布，所以构建了fc_mean,fc_std
        # 输出均值和标准差
        self.fc_mean = nn.Linear(hidden_dim, action_dim) # 均值层
        self.fc_std = nn.Linear(hidden_dim, action_dim) # 标准差
        
        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus() # 用于保证标准差为正数（需在 forward 中调用）
        
        # demo环境pendulum摇摆力度（-2，2）

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
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) #接着前面的hidden_dim
        self.fc3 = nn.Linear(hidden_dim, out_features=1) #输出特征，1维
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x)) #过第一层relu激活
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value

#经验回放，按理来说PPO不该用这个。这个类用来存放一些交互的经验
class ReplayMemory:
    def __init__(self, batch_size): #为什么经验池缺了td计算需要的next_state？ | td：时序差分
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.BATCH_SIZE = batch_size

    def add_memo(self, state, action, reward, value, done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)

    def sample(self):
        num_state = len(self.state_cap)
        #np.arange函数返回一个有终点和起点的固定步长的排列（可理解为等差数组）
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        memory_indicies = np.arange(num_state, dtype=np.int32) #shuffle
        np.random.shuffle(memory_indicies)
        batches = [memory_indicies[i:i+self.BATCH_SIZE] for i in batch_start_points]

        return np.array(self.state_cap), \
            np.array(self.action_cap), \
            np.array(self.reward_cap), \
            np.array(self.value_cap), \
            np.array(self.done_cap), \
            batches


    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []


#Actor-Critic网络和ReplayMemory会被统一封装在PPOAgent类中
class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):
        self.LR_ACTOR = 3e-4 # Actor网络学习率
        self.LR_CRITIC = 3e-4 # CRITIC网络学习率
        self.GAMMA = 0.99 #gamma为小于1的衰减因子，且随指数衰减，长期任务取0.99-0.999合适
        self.LAMBDA = 0.95 #GAE算法公式中的lambda，也是一个折扣衰减因子
        self.NUM_EPOCH = 10 #在update方法的for循环上使用
        self.EPSILON_CLIP = 0.2 #LossPPO_2 中使用的截断函数的epsilon取值
        # 对应公式：clip(P_theta(a^t|S_n^t).....)
        # 对应例子：参考学习对象小明来调整自己的策略，但是小明不能和自己差异过大
        # 因此要么使用散度补正的LossPPO_1，要么使用截断函数补正的LossPPO_2

        self.actor = Actor(state_dim, action_dim).to(device) #放到gpu上
        self.old_actor = Actor(state_dim, action_dim).to(device) #一样的神经网络，一样初始化
        self.critic = Critic(state_dim).to(device) #同样放到gpu
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR) #用Adam来调整actor网络参数
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC) #Adam最常见，无脑用了
        self.repaly_buffer = ReplayMemory(batch_size) #实例化




    def get_action(self, state):
        # 将state转化成tensor张量方便计算，并进行unsqueeze升维操作，最后移动到gpu上
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state) #通过前面实现的actor选择action
        value = self.critic.forward(state)
        # 由于这里action和value是神经网络计算得到的，我们要做detach操作，并返回成numpy数组形式
        # 以便后续处理操作。即：detach()分离梯度，cpu()转移至CPU，numpy()转换成numpy数组，[0]提取第一个标量
        # detach()用于将张量从计算图中分离，使其不再参与梯度计算。
        # 若张量仍关联梯度，后续操作（如与环境交互）会引发错误，且占用额外内存。
        # cpu()将张量从 GPU 显存转移到 CPU 内存。PyTorch 张量默认可能位于 GPU，而
        # NumPy 仅支持 CPU 数据。若张量已在 CPU 上，此操作无影响。
        # numpy()将 PyTorch 张量（Tensor）转换为 NumPy 数组（ndarray）。
        # 前提是：张量必须位于 CPU 上且已分离梯度（即需先执行 detach().cpu()）。
        # 为何使用 [0]？
        # 当输入是批处理数据（如 batch_size=32）时，可能需要遍历所有结果。
        # 若只需单样本结果（例如实时控制），则取 [0] 提取第一个样本。
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]


    def update(self):
        '''
        在 PyTorch 中，self.old_actor.load_state_dict(self.actor.state_dict())
        这行代码的作用是将当前策略网络（self.actor）的参数复制到旧策略网络
        （self.old_actor）中。这是强化学习算法（如 PPO、DDPG 等）中常见的
        关键操作，目的是冻结旧策略的参数用于后续计算（如重要性采样），同时
        允许当前策略继续更新。
        '''
        # 注意，self.old_actor = self.actor会导致 self.old_actor 和
        # self.actor 指向同一个对象，参数实时同步，失去冻结旧策略的意义。
        # 那样是错误的浅拷贝操作。正确做法是使用load_state_dict
        # 通过参数字典实现参数的深拷贝，确保两个网络的参数独立。
        # 在 PPO 中，通常在以下时刻同步参数：
        # -每次收集完一批新数据后。
        # -在多次策略更新（如 K 个 epochs）之前同步一次旧策略。
        self.old_actor.load_state_dict(self.actor.state_dict()) #老网络与现有网络参数同步, 此时ratio=1
        # Algorithm 1 PPO, Actor-Critic Style
        # Optimize surrogate L wrt theta, with K epochs and minibatch size M <= NT
        # theta_old = theta
        for epoch_i in range(self.NUM_EPOCH):
            # 取样整个memory
            memo_states, memo_actions, memo_rewards, memo_values, memo_dones, batches = self.repaly_buffer.sample()
            T = len(memo_rewards) # 用老的策略跑T步
            # 为GAE计算结果申请内存空间，跑T步，申请T个，类型选择精度没那么高的float32够用
            memo_advantages = np.zeros(T, dtype = np.float32)

            for t in range(T): # 请参阅论文公式(11), 公式(12)关于advatages的计算
                discount = 1 # 参阅(11)，衰减因子初始值为1，之后不断指数
                a_t = 0 # a_t即公式(11)要计算的值，GAE优势函数表达式

                for k in range(t, T-1): #计算公式(11)即GAE优势函数
                    reward_t = memo_rewards[k] #(12)中的r_t
                    next_state_value = memo_values[k+1] * (1-int(memo_dones[k])) #如果是最后一个状态，就没有next_state_value了
                    td_error = reward_t + self.GAMMA * next_state_value #用TD误差替代Q_theta(S, a)真实值，避免依赖完整轨迹Traj
                    value_st= memo_values[k]
                    delta_t = td_error - value_st #优势函数A_theta(S, a) = Q_theta(S, a) - V_theta(S)
                    # 优势函数A_theta(S, a) 的含义是在State S下，做出Action a，比其他动作能带来多少优势
                    delta_t *= discount #参考公式(11)，每次往后采样，距离当前状态越远，影响越小，需要乘上折扣因子
                    a_t += delta_t # 参考公式(11)，GAE=sum(delta_t*discount_t)
                    discount *= self.GAMMA * self.LAMBDA #discount衰减一个指数，(gamma*lamda)^n
                
                memo_advantages[t] = a_t #存储GAE计算结果
            
            # 已经计算好了T步对应的GAE值，准备求Pi(Pi_theta)，参考论文公式(6)与(7)
            # 求取新旧Pi的目的是计算重要性权重，用于重要性采样(Importance Sampling)
            # 重要性权重 = pi_new(a|s) / pi_old(a|s)
            with torch.no_grad(): # 这一步骤不需要跟踪梯度
                # advantages从np数组转张量并升维到二维，最后移动到gpu上
                # 为什么要升维到二维？
                memo_advantages_tensor = torch.tensor(memo_advantages).unsqueeze(1).to(device)
                memo_values_tensor = torch.tensor(memo_values).to(device) #本来就是二维，所以不升维了（为什么是二维？）
            #同样转换states和actions，为计算做准备
            memo_states_tensor = torch.FloatTensor(memo_states).to(device)
            memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)

            for batch in batches: # 1.遍历数据批次：将存储的经验数据分批次处理;
                with torch.no_grad():
                    # 2. 冻结旧策略：使用旧策略网络（self.old_actor）计算动作
                    # 分布参数（均值 old_mu 和标准差 old_sigma）;
                    # 获取第batch个状态下的旧策略网络正态分布对应的mu和sigma值
                    # old_mu：均值，决定分布的中心位置。
                    # old_sigma：标准差，控制动作的探索范围。
                    old_mu, old_sigma = self.old_actor(memo_states_tensor[batch])
                    # 3. 构建高斯分布：基于参数创建正态分布对象，表示旧策略的动作分布;
                    # Normal 是 PyTorch 的概率分布类，支持对数概率计算和采样。
                    old_pi = Normal(old_mu, old_sigma)
                # 计算old_pi的log对数，即当前批次动作的对数概率，用于后续计算重要性采样权重。
                batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch])

                mu, sigma = self.actor(memo_states_tensor[batch])
                pi = Normal(mu, sigma)
                batch_probs_tensor = pi.log_prob(memo_actions_tensor[batch])

                ratio = torch.exp(batch_probs_tensor - batch_old_probs_tensor)
                #surrogate?
                surr1 = ratio * memo_advantages_tensor[batch]
                surr2 = torch.clamp(ratio, 1-self.EPSILON_CLIP, 1+self.EPSILON_CLIP) * memo_advantages[batch]

                LossPPO2 = -torch.min(surr1, surr2).mean()

                actor_loss = LossPPO2

                # batch_returns 就是 Q = A + V, 我们曾用td_error来近似替代Q
                # 换句话说，计算真实回报：batch_returns = 优势 + 旧价值估计
                batch_returns = memo_advantages_tensor[batch] + memo_values_tensor[batch]
                #预测价值：self.critic（价值网络）对状态的新估计值。（？）
                batch_old_values = self.critic(memo_states_tensor[batch])
                
                # 最小化预测值与真实回报的均方误差
                critic_loss = nn.MSELoss()(batch_old_values, batch_returns) 

                #更新Actor网络参数
                self.actor_optimizer.zero_grad() # 清空优化器的历史梯度
                actor_loss.backward() # 反向传播
                self.actor_optimizer.step() # 根据梯度更新策略网络参数

                #更新Critic网络参数
                self.critic_optimizer.zero_grad() # 清空优化器的历史梯度
                critic_loss.backward() # 反向传播
                self.critic_optimizer.step() # 根据梯度更新策略网络参数

        self.repaly_buffer.clear_memo() # 清空经验池，下次update用新的经验
                

    def save_policy(self):
        torch.save(self.actor.state_dict(), f"ppo_policy_pendulum_v1.para")