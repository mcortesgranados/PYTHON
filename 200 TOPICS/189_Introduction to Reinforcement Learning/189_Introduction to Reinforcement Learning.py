import gym
import numpy as np
import tensorflow as tf

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=env.observation_space.shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MeanSquaredError())

# Train the model
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < 0.1:
            action = env.action_space.sample()  # Explore
        else:
            q_values = model.predict(np.expand_dims(state, axis=0))
            action = np.argmax(q_values)  # Exploit

        # Take action and observe next state and reward
        next_state, reward, done, _ = env.step(action)

        # Update the model
        target = reward + 0.99 * np.max(model.predict(np.expand_dims(next_state, axis=0)))
        target_vector = model.predict(np.expand_dims(state, axis=0))[0]
        target_vector[action] = target
        model.fit(np.expand_dims(state, axis=0), np.expand_dims(target_vector, axis=0), verbose=0)

        state = next_state

# Evaluate the trained agent
total_rewards = []
for _ in range(100):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(model.predict(np.expand_dims(state, axis=0)))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    total_rewards.append(total_reward)

print("Average total rewards over 100 episodes:", np.mean(total_rewards))
