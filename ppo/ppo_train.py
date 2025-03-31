import time
import os.path
import gymnasium as gym #trying to migrate from gym to gymnasium
import torch
import numpy as np
from ppo_agent import PPOAgent #todo

scenario = 'Pendulum-v1'
env = gym.make(scenario)

#保存模型的路径和文件格式
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + "/models/"
if not os.path.exists(model): #若文件夹不存在则新建，否则可能无法保存模型
    os.makedirs(model)
timestamp = time.strftime("%Y%m%d%H%M%S")

#超参设置
NUM_EPISODE = 3000
NUM_STEP = 200
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BATCH_SIZE = 25
UPDATE_INTERVAL = 50 #batc_size的两倍

#选定智能体
agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE) #todo

#初始化best_reward，用来在保存策略中判断后续是否应该保存模型
REWARD_BUFFER = np.empty(shape = NUM_EPISODE)
best_reward = -2000

for episode_i in range(NUM_EPISODE):
    state, others = env.reset() #初始化环境
    done = False
    episode_reward = 0

    for step_i in range(NUM_STEP):
        action, value = agent.get_action(state) #value可以一起计算出来
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        # Gym 从 0.26 版本开始引入 truncated 和 terminated 分离的终止标志：
        # terminated：环境因任务成功/失败终止（如游戏胜利或死亡）。
        # truncated：环境因外部条件终止（如步数耗尽）。
        # 本demo环境pendulum可能无truncated信号，所以done中不加处理
        # 从而覆盖了环境原本的终止信号
        done = True if (step_i+1) == NUM_STEP else False
        agent.replay_buffer.add_memo(state, action, reward, value, done) #agent获取经验，todo
        state = next_state #准备开始下一个循环

        if (step_i + 1) % UPDATE_INTERVAL == 0 or (step_i + 1) ==NUM_STEP:
            agent.update() # todo，更新自己的策略
    
    #保存策略，样例情况下，reward总是负数
    if episode_reward >= -100 and episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy()
        torch.save(agent.actor.state_dict(), model + f"ppo_actor_{timestamp}.pth")
        print(f"Best reward: {best_reward}")

    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode: {episode_i}, Reward: {round(episode_reward, 2)}") #小数点后两位
        
