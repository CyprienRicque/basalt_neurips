# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.

from argparse import ArgumentParser
import pickle
import time

import gym
import torch
import numpy as np
from tqdm import tqdm

from src.agent import MineRLAgent
from src.original_agent import MineRLAgent as MineRLAgent_original
from src.data_loader import DataLoader
from src.lib.tree_util import tree_map

import logging

LEVEL = logging.INFO


logger = logging.getLogger(__name__)
logger.setLevel(LEVEL)
console_handler = logging.StreamHandler()
console_handler.setLevel(LEVEL)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Originally this code was designed for a small dataset of ~20 demonstrations per task.
# The settings might not be the best for the full BASALT dataset (thousands of demonstrations).
# Use this flag to switch between the two settings

EPOCHS = 1
# Needs to be <= number of videos
BATCH_SIZE = 6
# Ideally more than batch size to create variation in datasets
# (otherwise, you will # get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 10
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

# Tuned with bit of trial and error
LEARNING_RATE = 0.000181
# OpenAI VPT BC weight decay
# WEIGHT_DECAY = 0.039428
WEIGHT_DECAY = 0.0
# KL loss to the original model was not used in OpenAI VPT
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0

MAX_BATCHES = int(1e9)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def behavioural_cloning_train_teacher(
    data_dir, in_model_original, in_model, in_weights_original, in_weights, out_weights, environment="MineRLBasaltFindCave-v0"
) -> None:
    """
    Fine-tune openai VPT on a dataset using behavioural cloning.
    :param in_weights_original:
    :param in_model_original:
    :param data_dir: directory containing the dataset
    :param in_model: model configuration file
    :param in_weights: model weights file
    :param out_weights: model weights file to save
    :param environment: environment to train on. All basalt environments have the same settings, so any of them works here
    :return: None
    """
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent_policy_kwargs_original, agent_pi_head_kwargs_original = load_model_parameters(in_model_original)

    # env = gym.make(environment)
    env = None

    agent = MineRLAgent(
        env,
        device=DEVICE,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(in_weights)

    # Create a copy which will have the original parameters
    original_agent = MineRLAgent_original(
        env,
        device=DEVICE,
        policy_kwargs=agent_policy_kwargs_original,
        pi_head_kwargs=agent_pi_head_kwargs_original,
    )
    original_agent.load_weights(in_weights_original)
    # env.close()

    policy = agent.policy
    original_policy = original_agent.policy

    # Freeze most params if using small dataset
    for param in policy.parameters():
        param.requires_grad = False
    # Unfreeze final layers
    trainable_parameters = []
    for param in policy.net.lastlayer.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    for param in policy.net.add_subtasks.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    for param in policy.pi_head.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)

    # Parameters taken from the OpenAI VPT paper
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
        dataset_max_size=400
    )

    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = torch.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Avg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
    for batch_i, (batch_images, batch_actions, batch_subtasks, batch_episode_id) in pbar:
        batch_loss = 0
        for image, action, subtasks, episode_id in zip(
            batch_images, batch_actions, batch_subtasks, batch_episode_id
        ):
            if image is None and action is None:
                # A work-item was done. Remove hidden state
                if episode_id in episode_hidden_states:
                    removed_hidden_state = episode_hidden_states.pop(episode_id)
                    del removed_hidden_state
                continue

            #print(action)
            agent_action = agent._env_action_to_agent(
                action, to_torch=True, check_if_null=True
            )
            if agent_action is None:
                continue

            subtasks_labels = agent.label_to_subtasks(subtasks, to_torch=True)

            agent_obs = agent._env_obs_to_agent({"pov": image, "subtasks": subtasks_labels})
            agent_obs_for_original = agent._env_obs_to_agent({"pov": image})

            if episode_id not in episode_hidden_states:
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                agent_obs, agent_state, dummy_first
            )

            with torch.no_grad():
                original_pi_distribution, _, _ = original_policy.get_output_for_observation(
                    agent_obs_for_original, agent_state, dummy_first
                )

            log_prob = policy.get_logprob_of_action(pi_distribution, agent_action)
            kl_div = policy.get_kl_of_action_dists(
                pi_distribution, original_pi_distribution
            )

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            # Finally, update the agent to increase the probability of the
            # taken action.
            # Remember to take mean over batch losses
            loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE
            batch_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            # time_since_start = time.time() - start_time
            # print(
            #     f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}"
            # )
            print("\n", end="")
            pbar.set_description(refresh=True, desc=f"Avg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0

        if batch_i % 1000 == 0:
            state_dict = policy.state_dict()
            torch.save(state_dict, out_weights)

        # if batch_i > MAX_BATCHES:
        #     break

    state_dict = policy.state_dict()
    torch.save(state_dict, out_weights)


if __name__ == "__main__":

    behavioural_cloning_train_teacher(
        data_dir="../basalt_neurips_data/MineRLBasaltMakeWaterfall-v0/",
        in_model_original="./data/VPT-models/2x.model",
        in_model="./data/VPT-models/2x.model",
        in_weights_original="./data/VPT-models/foundation-model-2x.weights",
        in_weights="./data/agent_st/foundation-model-tl-tt-2x.weights",
        out_weights="./data/agent_st/foundation-model-tl-bct-2x.weights_TMP",
        environment="MineRLBasaltFindCave-v0"
    )
