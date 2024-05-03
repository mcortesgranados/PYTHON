"""
06. Training deep reinforcement learning models for game playing or control systems.

In this example:

Imports: Import necessary libraries including PyTorch, OpenAI Gym, and other utilities.
Device Configuration: Set the device to GPU if available, else CPU.
Hyperparameters: Define hyperparameters such as discount factor, learning rate, etc.
Replay Buffer: Implement a replay buffer to store and sample experiences.
Q-Network: Define a simple DQN architecture.
Epsilon-Greedy Strategy: Implement an epsilon-greedy strategy for exploration.
Agent: Create an agent to interact with the environment.
Training Functions: Implement functions to compute Q-values, target Q-values, and train the DQN.
Main Function: Train the DQN agent to play CartPole.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import gym

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
gamma = 0.99  # Discount factor
batch_size = 64
lr = 1e-3  # Learning rate
eps_start = 1.0  # Initial epsilon value for epsilon-greedy exploration
eps_end = 0.01  # Final epsilon value
eps_decay = 0.995  # Epsilon decay rate
target_update = 10  # Update target network every 'target_update' episodes
num_episodes = 1000  # Number of episodes to train

# Define the Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define Epsilon-Greedy Strategy
class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * np.exp(-1. * current_step * self.decay)

# Define Agent
class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
    
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        
        if rate > random.random():
            return torch.tensor([random.randrange(self.num_actions)], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)

# Function to compute Q-value
def compute_q_val(policy_net, state, action):
    return policy_net(state).gather(dim=1, index=action.unsqueeze(-1))

# Function to compute target Q-value
def compute_target(policy_net, target_net, reward, next_state, done):
    target = reward + gamma * target_net(next_state).max(dim=1)[0].detach() * (1 - done)
    return target.unsqueeze(-1)

# Function to train the DQN
def train_dqn(policy_net, target_net, memory, optimizer):
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    
    q_values = compute_q_val(policy_net, state_batch, action_batch)
    target_values = compute_target(policy_net, target_net, reward_batch, next_state_batch, done=0)
    
    loss = nn.functional.smooth_l1_loss(q_values, target_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main function for training the DQN
def main():
    env = gym.make('CartPole-v1')
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    policy_net = DQN(num_inputs, num_actions).to(device)
    target_net = DQN(num_inputs, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    memory = ReplayBuffer(10000)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, num_actions, device)
    
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        
        for timestep in range(1000):  # Maximum 1000 timesteps per episode
            action = agent.select_action(state, policy_net)
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            
            memory.push(state, action, next_state, reward)
            state = next_state
            
            train_dqn(policy_net, target_net, memory, optimizer)
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        # Update the target network every 'target_update' episodes
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Print episode information
        if episode % 20 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
    
    env.close()

if __name__ == "__main__":
    main()
