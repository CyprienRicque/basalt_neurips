# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.

import pickle
import time

import gym
import torch
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.agent import MineRLAgent
from src.original_agent import MineRLAgent as MineRLAgent_original

from src.lib.tree_util import tree_map

import logging

import random

seed = 42

random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

EPOCHS = 1
# Needs to be <= number of videos
BATCH_SIZE = 4
# Ideally more than batch size to create variation in datasets (otherwise, you will get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 8
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

# Tuned with bit of trial and error
LEARNING_RATE = 0.000181
# OpenAI VPT BC weight decay : WEIGHT_DECAY = 0.039428
WEIGHT_DECAY = 0.0
# KL loss to the original model was not used in OpenAI VPT
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0

DATASET_MAX_SIZE = 400

MAX_BATCHES = int(1e9)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def setup_bc(in_model, in_model_original, in_weights, in_weights_original):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent_policy_kwargs_original, agent_pi_head_kwargs_original = load_model_parameters(in_model_original)
    # env = gym.make(environment)
    device = DEVICE
    agent = MineRLAgent(
        env=None,
        device=device,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    # print(f"0.1. {torch.cuda.memory_allocated(0)=}")

    agent.load_weights(in_weights)
    # print(f"0.2. {torch.cuda.memory_allocated(0)=}")

    # Create a copy which will have the original parameters
    original_agent = MineRLAgent_original(
        env=None,
        device=device,
        policy_kwargs=agent_policy_kwargs_original,
        pi_head_kwargs=agent_pi_head_kwargs_original,
    )
    # print(f"0.3. {torch.cuda.memory_allocated(0)=}")

    original_agent.load_weights(in_weights_original)
    # print(f"0.4. {torch.cuda.memory_allocated(0)=}")

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
    # print(f"0.5. {torch.cuda.memory_allocated(0)=}")

    # Parameters taken from the OpenAI VPT paper
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # print(f"0.6. {torch.cuda.memory_allocated(0)=}")
    policy.initial_state(N_WORKERS, 1)
    original_policy.initial_state(N_WORKERS, 1)
    policy.to(device)
    original_policy.to(device)

    return device, optimizer, original_policy, policy, trainable_parameters, agent, original_agent


def init(in_model, in_model_original, in_weights, in_weights_original):
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
    return agent, optimizer, original_policy, policy, trainable_parameters


total = 0
count = 0


def get_item_size(item) -> int:
    if item is None:
        return 0
    size = 0
    if isinstance(item, tuple) or isinstance(item, list):
        for i in item:
            size += get_item_size(i)
    elif isinstance(item, dict):
        for i in item:
            size += get_item_size(item[i])
    elif isinstance(item, torch.Tensor):
        size += item.element_size() * item.nelement()
        if item.device.__str__() != 'cuda:0':
            print('device is', item.device)
            exit(1)
    else:
        print("no size for", item)
    return size


def get_item_shape(item) -> str:
    if item is None:
        return "no"
    shape = 0
    if isinstance(item, tuple) or isinstance(item, list):
        for i in item:
            shape = f"{shape}, {get_item_shape(i)}"
    elif isinstance(item, dict):
        for i in item:
            shape = f"{shape}, {get_item_shape(item[i])}"
    elif isinstance(item, torch.Tensor):
        shape = f"{shape}, {item.shape}"
        if item.device.__str__() != 'cuda:0':
            print('device is', item.device)
            exit(1)
    return shape


def model_forward(agent_action, agent_obs, agent_obs_for_original, ids, batch_loss, dummy_first,
                  episode_hidden_states, episode_id, original_policy, policy):
    global total, count

    # print(f"{agent_obs=}")

    # Convert the PyTorch tensor to a NumPy array
    # image_numpy = agent_obs['img'].detach().cpu().numpy()
    # print(image_numpy)

    # Convert the NumPy array to a PIL image
    # plt.imshow(image_numpy[0], interpolation='nearest')
    # plt.show()
    # exit(0)

    # print("action in", count, agent_action)
    # print("obs in", count, agent_obs)

    # print(f"{agent_obs=}")
    # print(f"{agent_state=}")
    # print(f"{dummy_first=}")

    # print(f"{agent_obs['img'].element_size() * agent_obs['img'].nelement()=}")
    # print(f"{agent_obs['subtasks'].element_size() * agent_obs['subtasks'].nelement()=}")

    # print(f"{len(agent_state[0][1])=}")

    # total_size = get_item_size(agent_action) + get_item_size(agent_obs) + get_item_size(agent_obs_for_original) + \
    #              get_item_size(agent_state) + get_item_size(dummy_first)
    # print(f"{total_size=}")

    # print(f"{get_item_size(agent_action)=}")
    # print(f"{get_item_size(agent_obs)=}")
    # print(f"{get_item_size(agent_obs_for_original)=}")
    # print(f"{get_item_size(agent_state)=}")
    # print(f"{get_item_size(dummy_first)=}")

    # print(f"{get_item_shape(agent_action)=}")
    # print(f"{get_item_shape(agent_obs)=}")
    # print(f"{get_item_shape(agent_obs_for_original)=}")
    # print(f"{get_item_shape(agent_state)=}")
    # print(f"{get_item_shape(dummy_first)=}")

    # print(f"mf. {torch.cuda.memory_allocated(0)=}")

    # torch.cuda.synchronize()
    # print(f"{ids=}")
    pi_distribution, _ = policy.get_output_for_observation(agent_obs, ids, dummy_first)

    with torch.no_grad():
        original_pi_distribution, _ = original_policy.get_output_for_observation(agent_obs_for_original, ids,
                                                                                 dummy_first)

    log_prob = policy.get_logprob_of_action(pi_distribution, agent_action)
    kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)
    loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE
    batch_loss += loss.item()
    loss.backward()

    # print(f"3. {torch.cuda.memory_allocated(0)=}")

    # 0.0120 vs 0.01539
    # 0.01910 vs 0.026842
    # print("model_forward.", end - start)

    # total += (end - start)
    # count += 1
    # print("mean:", total / count)

    # if end - start < 0.019:
    #     exit(2)
    #     print(f"{agent_action=}")
    #     print(f"{agent_obs=}")
    #     print(f"{agent_obs_for_original=}")
    #     print(f"{agent_state=}")
    #     print(f"{episode_id=}")
    return batch_loss


def model_forward_batch(agent_action, agent_obs, agent_obs_for_original, dummy_first,
                        episode_id, worker_ids, original_policy, policy):
    pi_distribution, _ = policy.get_output_for_observation(agent_obs, worker_ids, dummy_first)

    with torch.no_grad():
        original_pi_distribution, _ = original_policy.get_output_for_observation(agent_obs_for_original, worker_ids, dummy_first)

    # print(f"{pi_distribution['camera'].shape=} {pi_distribution['buttons'].shape=} {agent_action=}")
    agent_action = {key: torch.tensor([i[key].item() for i in agent_action]).to(device=DEVICE, non_blocking=True) for key in agent_action[0]}

    log_prob = policy.get_logprob_of_action(pi_distribution, agent_action)
    # print(f'{log_prob=}')  # log_prob=tensor([-2.3904], device='cuda:0', grad_fn=<SelectBackward0>)
    kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)

    # print(f'{kl_div=}')  # kl_div=tensor([[[1.6907]]], device='cuda:0', grad_fn=<AddBackward0>)

    loss = torch.mean(-log_prob + KL_LOSS_WEIGHT * kl_div.squeeze())
    loss.backward()

    return loss.item()


