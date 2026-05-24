import gymnasium as gym
import random
import numpy as np
import torch 
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt
import time


#---Neural Network---
#Contains 4 layers
#Uses ReLU activation function

class Model(nn.Module):
    def __init__(self, Hidden):
        super().__init__()
        self.layer1 = nn.Linear(8, Hidden)
        self.layer2 = nn.Linear(Hidden, Hidden)
        self.layer3 = nn.Linear(Hidden, 4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
    

#---Buffer (sampling)---
#
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
    


# region ---Hyperparameters---
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
lr = 0.0005 #learning rate
target_update_freq = 100  # sync target network every 100 steps

# Create everything
model = Model(Hidden = 64)
target_model = Model(Hidden=64)

target_model.load_state_dict(model.state_dict())  # start as a copy

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()
buffer = ReplayBuffer(capacity=20000)
env = gym.make("LunarLander-v3")  # no render during training, faster

step_count = 0
num_episodes = 1500
reward_history = []
best_avg = -np.inf
window = 50
# endregion
        



#---Training Loop---
for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    #epsilon greedy allows for exploration
    while not done:
        if random.random() < epsilon:   
            action = env.action_space.sample()
        else:
            q_values = model(torch.tensor(obs, dtype=torch.float32))
            action = torch.argmax(q_values).item()
        
        #Take the action (from NN)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        step_count += 1

        #Train when there's enough samples
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            #current q values
            q_values = model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            best_actions = model(next_states).argmax(dim=1)
            max_next_q = target_model(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            targets = (rewards + gamma * max_next_q * (1 - dones)).detach()

            #Update
            MSE = loss(q_values, targets)
            optimizer.zero_grad()
            MSE.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #clipping prevents gradient descent overshoot
            optimizer.step()
    
    
        if step_count % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())


    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_history.append(total_reward)

    if episode % 20 == 0:
        avg = np.mean(reward_history[-20:])
        print(f"Episode {episode}, Avg Reward: {avg:.1f}, Epsilon: {epsilon:.3f}")
        if avg > best_avg:
            best_avg = avg
            torch.save(model, "/Users/michael/Desktop/Summer Project/NN/best_lunar_lander.pth")




# region Size, Latency (Feasibility in a real rocket)
model.eval()
model_parameters = list(model.parameters()) + list(model.buffers())
size = 0
for t in model_parameters:
    size += t.numel() * t.element_size()
print(f'[Model Size] = {size} bytes, ~ {size/(1024**2):.3f}MB')
# endregion



x_base = torch.randn(1, 8, device= "cpu")
batch_size1 = 64
x1 = x_base.expand(batch_size1, -1).contiguous()
batch_size2 = 8
x2 = x_base.expand(batch_size2, -1).contiguous()


def Latency(model, x, iters):
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(iters):
            model(x)
        elapsed = time.perf_counter() - start
        latency = elapsed/iters
    
    return latency

latency_x1 = Latency(model, x1, 50)
latency_x2 = Latency(model, x2, 50)

print(latency_x1, latency_x2)



    


# Watch the trained agent
model = torch.load("/Users/michael/Desktop/Summer Project/NN/best_lunar_lander.pth", weights_only = False)

env = gym.make("LunarLander-v3", render_mode="human")

for _ in range(5):  # watch 5 episodes
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            q_values = model(torch.tensor(obs, dtype=torch.float32))
            action = torch.argmax(q_values).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Reward: {total_reward:.1f}")

env.close()


smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
plt.plot(smoothed)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Double DQN on LunarLander")
plt.show()






