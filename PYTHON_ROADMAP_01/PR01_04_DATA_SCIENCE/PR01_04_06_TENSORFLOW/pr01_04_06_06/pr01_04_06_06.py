# %%                                # This is for Jupyter-style cell execution (optional)
"""
06. Training deep reinforcement learning models for game playing or control systems.
Dependencies: pip install gym torch numpy
"""

# Import necessary libraries
import gym                               # Imports the OpenAI Gym library to simulate environments
import random                            # Imports Python's random module for random number generation
import numpy as np                       # Imports NumPy for numerical operations on arrays
from collections import deque            # Imports deque for efficiently managing the replay buffer
import torch                             # Imports PyTorch for deep learning
import torch.nn as nn                    # Imports the neural network module from PyTorch
import torch.optim as optim              # Imports optimizers from PyTorch

# Set the computation device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Chooses GPU if available, otherwise CPU

# Define the Deep Q-Network class
class DQN(nn.Module):                                   # Defines a neural network class for DQN, inheriting from nn.Module
    def __init__(self, state_size, action_size):        # Constructor with input and output dimensions
        super(DQN, self).__init__()                     # Calls the parent class constructor
        self.fc1 = nn.Linear(state_size, 24)            # First fully connected layer (input to 24 hidden units)
        self.fc2 = nn.Linear(24, 24)                    # Second fully connected layer (24 to 24 hidden units)
        self.fc3 = nn.Linear(24, action_size)           # Output layer (24 to number of actions)

    def forward(self, x):                               # Forward pass through the network
        x = torch.relu(self.fc1(x))                     # Apply ReLU activation to the first layer
        x = torch.relu(self.fc2(x))                     # Apply ReLU activation to the second layer
        return self.fc3(x)                              # Return output layer (Q-values for each action)

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode=None)         # Initialize the CartPole environment (v1 has a 500 step limit)

# Extract state and action dimensions from the environment
state_size = env.observation_space.shape[0]             # Get the number of state features (4 for CartPole)
action_size = env.action_space.n                        # Get the number of possible actions (2 for CartPole)

# Define hyperparameters
learning_rate = 0.001                                   # Learning rate for the optimizer
gamma = 0.99                                            # Discount factor for future rewards
epsilon = 1.0                                           # Initial value of epsilon for exploration
epsilon_min = 0.01                                      # Minimum value epsilon can decay to
epsilon_decay = 0.995                                   # Rate at which epsilon decays per episode
batch_size = 64                                         # Size of each training batch
memory = deque(maxlen=2000)                             # Replay memory to store past experiences (with a max length)

# Initialize the model and optimizer
model = DQN(state_size, action_size).to(device)         # Create the DQN model and move it to the selected device
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Use Adam optimizer with specified learning rate
loss_fn = nn.MSELoss()                                  # Mean Squared Error loss function for regression of Q-values

# Define epsilon-greedy action selection
def act(state, epsilon):                                # Function to select an action using epsilon-greedy strategy
    if np.random.rand() <= epsilon:                     # With probability epsilon, choose a random action
        return random.randrange(action_size)            # Return a random action from action space
    state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Convert state to torch tensor and add batch dimension
    q_values = model(state)                             # Compute Q-values using the current model
    return torch.argmax(q_values).item()                # Return the index of the action with the highest Q-value

# Define experience replay training
def replay():                                           # Function to sample a batch and train the model
    if len(memory) < batch_size:                        # Skip training if there aren't enough samples in memory
        return                                          # Exit the function early

    minibatch = random.sample(memory, batch_size)       # Randomly sample a batch from replay memory
    states, actions, rewards, next_states, dones = zip(*minibatch)  # Unpack batch into separate components

    states = torch.FloatTensor(states).to(device)       # Convert states to torch tensor and move to device
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # Convert actions and reshape for gather()
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # Convert rewards and reshape
    next_states = torch.FloatTensor(next_states).to(device)       # Convert next states
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)      # Convert done flags and reshape

    q_values = model(states).gather(1, actions)         # Get Q-values of taken actions using gather()

    next_q_values = model(next_states).max(1)[0].unsqueeze(1)  # Get maximum predicted Q-values for next states
    target = rewards + (gamma * next_q_values * (1 - dones))   # Compute target Q-values using Bellman equation

    loss = loss_fn(q_values, target)                    # Compute loss between predicted and target Q-values

    optimizer.zero_grad()                               # Zero the gradients before backpropagation
    loss.backward()                                     # Compute gradients using backpropagation
    optimizer.step()                                    # Update model parameters using the optimizer

# Set number of training episodes
episodes = 500                                          # Total number of episodes to train the agent

# Start training loop
for e in range(episodes):                               # Loop through each episode
    state, _ = env.reset()                              # Reset environment to get the initial state (new gym API)
    state = np.array(state)                             # Convert the initial state to a NumPy array
    total_reward = 0                                    # Initialize total reward for this episode

    for time_t in range(500):                           # Limit each episode to a max of 500 steps
        action = act(state, epsilon)                    # Select action using epsilon-greedy policy
        next_state, reward, terminated, truncated, _ = env.step(action)  # Take action and observe result
        done = terminated or truncated                  # Check if episode is done (either terminated or truncated)
        next_state = np.array(next_state)               # Convert next state to NumPy array
        memory.append((state, action, reward, next_state, done))  # Store experience in memory
        state = next_state                              # Update current state to the next state
        total_reward += reward                          # Accumulate the reward for this episode

        replay()                                        # Call the replay function to train the model

        if done:                                        # Break the loop if the episode has ended
            break                                       # Exit the time step loop

    epsilon = max(epsilon_min, epsilon_decay * epsilon)  # Reduce epsilon after each episode (with minimum limit)
    print(f"Episode {e+1}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.2f}")  # Print progress

# Training complete
print("Training completed!")                            # Print message after all episodes are done
