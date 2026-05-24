import gymnasium as gym
import torch as torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

"""
PPO (Proximal Policy Optimization) for LunarLander-v3 continuous.
"""


#========== NEURAL NETWORK ==========
# There are 2 neural networks. 
# The 1st is the "Actor": Given a state, it outputs 2 Gaussians (1 for thrust, 1 for gimbal). This tells the rocket what to do.
# The 2nd is the "Critic": Given a state, it outputs a number. This represents how good the rocket's current state is.

class Actor(nn.Module):
    def __init__(self, inp, hidden, outp):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear (hidden, outp)
        self.relu = nn.ReLU()
        self.log_std = nn.Parameter(torch.zeros(outp)) 

    def forward(self, mean):
        mean = self.fc1(mean)
        mean = self.relu(mean)
        mean = self.fc2(mean)
        mean = self.relu(mean)
        mean = self.fc3(mean)

        stds = self.log_std.exp() 
        dist = torch.distributions.Normal(mean, stds)  
        return dist 
        

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
        



#========== Create a buffer to store experiences ==========
#Buffer is a list of lists
class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    


#========== Hyperparameters ==========
actor_lr = 0.0003
critic_lr = 0.0003
steps = 2048
gamma = 0.99
num_epochs = 2
num_iterations = 1500
gae_lambda = 0.95  #standard

best_avg = -np.inf


#Make everything
env = gym.make("LunarLander-v3", continuous = True)
actor = Actor(8, 128, 2)
critic = Critic(8, 128, 1)
buffer = Buffer()

actor_optim = torch.optim.Adam(actor.parameters(), lr = actor_lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr = critic_lr)





# ========== TRAINING LOOP ==========
"""
Main structure:

Training (1500 iterations)
└── Each iteration:
    ├── Collect 2048 steps (5-10 episodes of gameplay experience). An episode ends when rocket crashes/timeouts
    ├── Compute advantages
    ├── Train for 2 epochs
    │   └── Each epoch: 32 mini-batches of 64 experiences
    │       └── Each mini-batch: one gradient update
    └── Clear buffer

"""
reward_history = []
for iteration in range(num_iterations):
    step_count = 0

    # decay learning rate
    lr = actor_lr * (1 - iteration / num_iterations)
    for param_group in actor_optim.param_groups:
        param_group['lr'] = lr
    for param_group in critic_optim.param_groups:
        param_group['lr'] = lr


    while step_count<steps:
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done and step_count < steps:
            #Compute action and value
            dists = actor(torch.tensor(obs, dtype = torch.float32))
            sample = dists.sample()
            clamped_sample = sample.clamp(-1, 1)
            log_prob = dists.log_prob(sample).sum() #Sum since they're log probabilities
            critic_output = critic(torch.tensor(obs, dtype = torch.float32)).squeeze()

            #Take action, store experience into buffer
            next_obs, reward, terminated, truncated, _ = env.step(clamped_sample.detach().numpy())
            done = terminated or truncated
            episode_reward += reward
            if done:
                reward_history.append(episode_reward)
            buffer.store(state = obs, action = sample, log_prob = log_prob, reward = reward,
                        value = critic_output, done = done)
            obs = next_obs
            step_count +=1


    #Computing advantages:
    #Uses GAE (Generalized Advantage Estimation)

    advantages = []
    gae = 0

    for i in range(len(buffer.states) - 1, -1, -1):
        if buffer.dones[i]:
            future_value = 0
            gae = 0
        elif i == len(buffer.states) - 1:
            future_value = critic(torch.tensor(buffer.states[-1], dtype=torch.float32)).squeeze().detach()
        else:
            future_value = buffer.values[i + 1].detach()
        
        delta = buffer.rewards[i] + (gamma * future_value) - buffer.values[i]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)



    #Normalize advantages 
    #Forces policy to differentiate good and bad actions (good = pos, bad = neg)
    advantages = torch.stack(advantages).detach()
    values = torch.stack(buffer.values).detach()
    returns = advantages + values

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)



    #Convert buffer items and advantages to Pytorch tensors
    states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
    actions = torch.stack(buffer.actions).detach()
    old_log_probs = torch.stack(buffer.log_probs).detach()
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
    dones = torch.tensor(buffer.dones, dtype=torch.float32)
   



    #From the buffer of 2048 experiences, sample 32 batches of 64 (more efficient)
    dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
    loader = DataLoader(dataset, batch_size = 64, shuffle=True)


    for epoch in range(num_epochs):
        for batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns in loader:

            #Compute Actor loss
            #The mechanism is called "Proximal Surrogate Objective". It prevents large, destabilizing updates to a policy
            #Compares new and old probability of an action occuring.
            #Makes good actions more probable, bad actions less probable
            new_dist = actor(batch_states)
            new_log_prob = new_dist.log_prob(batch_actions).sum(-1)
            ratio = torch.exp(new_log_prob - batch_log_probs)
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
            actor_loss = (-torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages)).mean()
            
            entropy = new_dist.entropy().mean()
            actor_loss = actor_loss - 0.01 * entropy #Avoids gaussian spread getting too small too quick

            #Compute Critic loss (Mean Squared Loss)
            new_value = critic(batch_states).squeeze()
            critic_loss = ((new_value - batch_returns)**2).mean()

            #Back propagation + updating neural network parameters
            #Clipping prevents gradient from changing too much in a single backward pass
            actor_optim.zero_grad()
            critic_optim.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5) 
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)

            actor_optim.step()
            critic_optim.step()


    episodes = sum(buffer.dones)
    buffer.clear() #Clear all 2048 experiences



    #The following is for analysis/viewing
    if iteration % 10 == 0:
        avg = np.mean(reward_history[-10:]) if len(reward_history) > 0 else 0
        print(f"Iteration {iteration}")
        print(f"  Avg Reward: {avg:.1f}")
        print(f"  log_std: {actor.log_std.data}")
        print(f"  Critic loss: {critic_loss.item():.4f}")
        print(f"  Actor loss: {actor_loss.item():.4f}")
        print(f"  Ratio min/max: {ratio.min().item():.3f} / {ratio.max().item():.3f}")
        print(f"  Episodes this iteration: {episodes}")

        #Save only the best model
        if avg > best_avg:
            best_avg = avg
            torch.save(actor, "/Users/michael/Desktop/Summer Project/NN/actor.pth")
            torch.save(critic, "/Users/michael/Desktop/Summer Project/NN/critic.pth")







        









    





        
        

























    




    

        
