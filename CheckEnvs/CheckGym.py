# 如果import gym，输出的环境是0.26.2版本
# 由于openAi已经废弃了gym转向gymnasium fork
# 为了保持与新版本numpy的兼容性，必须从gym迁移到gymnasium环境
# 为此我们需要import gymnasium as gym 来尽可能少的改动基于gym环境编写的代码
# 同时向gymnasium环境迁移
import gymnasium as gym 
from gym import envs

print(f"gym_version = {gym.__version__}")

# 获取所有环境ID列表
#env_ids = list(envs.registry.keys())

#print(f'There are {len(env_ids)} envs in gym')
#print(env_ids)