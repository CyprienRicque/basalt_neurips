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
from src.behavioural_cloning_common import DEVICE, LEARNING_RATE, WEIGHT_DECAY, N_WORKERS, \
    BATCH_SIZE, EPOCHS, LOSS_REPORT_RATE, DATASET_MAX_SIZE, setup_bc, model_forward
from src.behavioural_cloning_common import KL_LOSS_WEIGHT, MAX_GRAD_NORM
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


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def behavioural_cloning_train_teacher(
        data_dir, in_model_original, in_model, in_weights_original, in_weights, out_weights,
        environment="MineRLBasaltFindCave-v0"
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
    print(f"0. {torch.cuda.memory_allocated(0)=}")

    device, optimizer, original_policy, policy, trainable_parameters, agent, original_agent = setup_bc(in_model,
                                                                                                       in_model_original,
                                                                                                       in_weights,
                                                                                                       in_weights_original)

    # print(f"1. {torch.cuda.memory_allocated(0)=}")
    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
        dataset_max_size=DATASET_MAX_SIZE
    )
    # print(f"dl. {torch.cuda.memory_allocated(0)=}")

    episode_hidden_states = {}
    dummy_first = torch.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Avg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
    # print(f"b1. {torch.cuda.memory_allocated(0)=}")

    for batch_i, (batch_images, batch_actions, batch_subtasks, batch_episode_id, worker_ids) in pbar:
        batch_loss = 0
        # print(f"b2. {torch.cuda.memory_allocated(0)=}")

        for image, action, subtasks, episode_id, worker_id in zip(
                batch_images, batch_actions, batch_subtasks, batch_episode_id, worker_ids
        ):

            if image is None and action is None:
                # A work-item was done. Remove hidden state
                if episode_id in episode_hidden_states:
                    removed_hidden_state = episode_hidden_states.pop(episode_id)
                    del removed_hidden_state
                continue
            # print(f"l1. {torch.cuda.memory_allocated(0)=}")

            agent_action = agent._env_action_to_agent(
                action, to_torch=True, check_if_null=True
            )

            if agent_action is None:
                continue

            subtasks_labels = agent.label_to_subtasks(subtasks, to_torch=True)

            agent_obs = agent._env_obs_to_agent({"pov": image, "subtasks": subtasks_labels})
            agent_obs_for_original = agent._env_obs_to_agent({"pov": image})

            # Model
            batch_loss = model_forward(agent_action, agent_obs, agent_obs_for_original,
                                       torch.tensor([worker_id], device=DEVICE), batch_loss,
                                       dummy_first, episode_hidden_states, episode_id, original_policy, policy)

        torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            print("\n", end="")
            pbar.set_description(refresh=True, desc=f"Avg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0

        if batch_i % 1000 == 0:
            state_dict = policy.state_dict()
            torch.save(state_dict, out_weights)

        # end = time.time(); print("4.", end - start); start = time.time()

    state_dict = policy.state_dict()
    torch.save(state_dict, out_weights)


if __name__ == "__main__":
    behavioural_cloning_train_teacher(
        data_dir="../basalt_neurips_data/MineRLBasaltMakeWaterfall-v0",
        in_model_original="./data/VPT-models/2x.model",
        in_model="./data/VPT-models/2x.model",
        in_weights_original="./data/VPT-models/foundation-model-2x.weights",
        in_weights="./data/agent_st/foundation-model-tl-tt-2x.weights",
        out_weights="./data/agent_st/foundation-model-tl-bct-2x.weights_TMP",
        environment="MineRLBasaltFindCave-v0"
    )
