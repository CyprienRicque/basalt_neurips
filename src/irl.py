import torch

import torch
import torch.nn as nn


import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class IRLModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(IRLModel, self).__init__()

        # Define the reward network
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # Compute the reward for the given state and action
        return self.reward_net(torch.cat([state, action], dim=1))


def train_IRL(expert_demos, policy_fn, irl_model, optimizer, num_iterations=1000):
    # expert_demos: a list of expert demonstrations, each represented as a tuple of (state, action) pairs
    # policy_fn: a function that takes in a state and returns the action to be taken in that state
    # irl_model: an instance of the IRLModel class
    # optimizer: an instance of a PyTorch optimizer (e.g. Adam, SGD)

    for iteration in range(num_iterations):
        # Compute the loss for the current iteration
        loss = IRL_loss(expert_demos, irl_model, policy_fn)

        # Backpropagate the error and update the model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the current loss
        print(f'Iteration {iteration}: Loss = {loss.item():.4f}')

    # Return the trained IRL model
    return irl_model


import torch


def IRL_loss(expert_demos, reward_fn, policy_fn):
    # expert_demos: a list of expert demonstrations, each represented as a tuple of (state, action) pairs
    # reward_fn: a function that takes in a state and action and returns the reward for that state-action pair
    # policy_fn: a function that takes in a state and returns the action to be taken in that state

    # Compute the expected reward for each expert demonstration
    expected_rewards = []
    for demo in expert_demos:
        expected_reward = 0
        for state, action in demo:
            expected_reward += reward_fn(state, action)
        expected_rewards.append(expected_reward)

    # Compute the log likelihood of the expert demonstrations under the policy_fn
    log_likelihood = 0
    for i, demo in enumerate(expert_demos):
        for state, action in demo:
            log_likelihood += torch.log(policy_fn(state)[action]) - expected_rewards[i]

    # Return the negative log likelihood as the loss
    return -log_likelihood



import torch


def MaxEnt_IRL_loss(expert_demos, reward_fn, policy_fn, entropy_coefficient):
    log_likelihood = 0

    for demo in expert_demos:
        expected_reward = 0
        for state, action in demo:
            expected_reward += reward_fn(state, action)
            log_likelihood += torch.log(policy_fn(state)[action]) - expected_reward

    states, actions = zip(*expert_demos)
    entropy = -torch.mean(torch.sum(policy_fn(states) * torch.log(policy_fn(states)), dim=1))
    loss = -log_likelihood + entropy_coefficient * entropy
    return loss
