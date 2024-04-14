# Hidden Markov Models (HMMs) are a class of probabilistic graphical models used to model sequential data, where the underlying system is assumed to be a Markov
# process with hidden states. HMMs are widely used in various fields, including speech recognition, natural language processing, bioinformatics, and finance.

# Here's a brief overview of the components of an HMM:

# Hidden States: The system being modeled is assumed to have a set of hidden states that are not directly observable. Each hidden state represents a 
# specific configuration or condition of the system.

# Observations: At each time step, an observation is emitted from the system based on its current hidden state. Observations are the only part of the model 
# that is directly observable.

# Transition Probabilities: HMMs assume that the system transitions between hidden states according to certain probabilities. These transition probabilities 
# determine the likelihood of transitioning from one hidden state to another.

# Emission Probabilities: Each hidden state has associated emission probabilities that govern the likelihood of emitting different observations when in that state.

# Initial State Distribution: HMMs also have an initial state distribution that specifies the probabilities of starting in each hidden state.

# The key characteristic of HMMs is that they are generative models, meaning they can generate sequences of observations by sampling from the model. 
# They can also be used for various tasks, including sequence prediction, sequence labeling, and sequence generation.

# Here's a high-level overview of how HMMs work:

# Initialization: Initialize the HMM parameters, including transition probabilities, emission probabilities, and initial state distribution.

# Forward Algorithm: Given a sequence of observations, compute the probability of observing that sequence given the model parameters. 
# This is done using the forward algorithm, which efficiently computes the probability of a sequence using dynamic programming.

# Viterbi Algorithm: Given a sequence of observations, find the most likely sequence of hidden states that could have generated the observations. 
# This is done using the Viterbi algorithm, which efficiently finds the most likely path through the hidden states.

# Training: Given a set of training data (sequences of observations), adjust the model parameters to maximize the likelihood of the training data. 
# This is typically done using the Expectation-Maximization (EM) algorithm or Baum-Welch algorithm.

# Inference: Given a trained model and a new sequence of observations, perform inference to predict hidden states, compute probabilities, or 
# generate new sequences.

# HMMs have been successfully applied to various real-world problems, including speech recognition (e.g., recognizing phonemes in speech signals), 
# part-of-speech tagging in natural language processing, and predicting financial time series data.

from hmmlearn import hmm
import numpy as np

# Define the HMM model parameters
n_hidden_states = 2
n_observations = 3  # Number of possible observations
model = hmm.MultinomialHMM(n_components=n_hidden_states, n_iter=100)

# Define the model parameters
model.startprob_ = np.array([0.6, 0.4])  # Initial state distribution
model.transmat_ = np.array([[0.7, 0.3],  # Transition probabilities
                            [0.4, 0.6]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5],  # Emission probabilities
                                [0.6, 0.3, 0.1]])

# Generate a sequence of observations
obs_seq = np.array([[0, 1, 2, 0, 2]]).T

# Fit the model to the data
model.fit(obs_seq)

# Predict the most likely sequence of hidden states
hidden_states = model.predict(obs_seq)

# Print the predicted hidden states
print("Predicted Hidden States:", hidden_states)
