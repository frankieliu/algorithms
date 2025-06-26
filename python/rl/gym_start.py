# https://gymnasium.farama.org/introduction/basic_usage/
#%%
import gymnasium as gym
# import torch
from torch import nn
#%%
env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()
#%%
# sample run
over = False
while not over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    over = terminated or truncated
env.close()
#%%
class policy(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layers = nn.Sequential(
            self.Linear(input, hidden),
            self.ReLU(),
            self.Linear(hidden, output),
            self.ReLU()
        )
    def forward(self, x):
        pass

class agent():
    model = env.action_space.n