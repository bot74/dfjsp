import gym
from gym import envs

# 获取所有环境ID列表
env_ids = list(envs.registry.keys())

print(f'There are {len(env_ids)} envs in gym')
print(env_ids)