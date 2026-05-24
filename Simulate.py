import gymnasium as gym
import random
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio


class Actor(nn.Module):
    def __init__(self, inp, hidden, outp):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear (hidden, outp)
        self.relu = nn.ReLU()
        self.log_std = nn.Parameter(torch.zeros(2)) 

    def forward(self, mean):
        mean = self.fc1(mean)
        mean = self.relu(mean)
        mean = self.fc2(mean)
        mean = self.relu(mean)
        mean = self.fc3(mean)

        stds = self.log_std.exp() 
        dist = torch.distributions.Normal(mean, stds)  
        return dist #Outputs 2 Gaussians (One for thrust, one for gimbal)
        
class Critic(nn.Module):
    def __init__(self, inp, hidden, outp):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear (hidden, outp)
        self.relu = nn.ReLU()
        

    def forward(self, value):
        value = self.fc1(value)
        value = self.relu(value)
        value = self.fc2(value)
        value = self.relu(value)
        value = self.fc3(value)
        return value 
        #Outputs a number. Represents the VALUE of a state (how good it is)


actor = torch.load("/Users/michael/Desktop/Summer Project/NN/actor.pth", weights_only = False)


frames = []
env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
obs, _ = env.reset()
done = False
while not done:
    with torch.no_grad():
        action = actor(torch.tensor(obs, dtype=torch.float32)).mean
    obs, reward, terminated, truncated, _ = env.step(action.numpy())
    frames.append(env.render())
    done = terminated or truncated

imageio.mimsave("landing.gif", frames, fps=30)


