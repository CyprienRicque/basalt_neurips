from argparse import ArgumentParser
import pickle
import time

import gym
import torch
import numpy as np
from tqdm import tqdm

from src.agent import MineRLAgent
from src.behavioural_cloning_common import setup_bc, BATCH_SIZE, N_WORKERS, EPOCHS, LOSS_REPORT_RATE, \
    MAX_GRAD_NORM, DATASET_MAX_SIZE, KL_LOSS_WEIGHT, model_forward, model_forward_batch
from src.original_agent import MineRLAgent as MineRLAgent_original
from src.data_loader_pp import DataLoader
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


def behavioural_cloning_train_teacher_batch(
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
    # print(f"0. {torch.cuda.memory_allocated(0)=}")

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
        dataset_max_size=DATASET_MAX_SIZE,
        device=device
    )
    # print(f"dl. {torch.cuda.memory_allocated(0)=}")

    episode_hidden_states = {}
    dummy_first = torch.from_numpy(np.array([False for _ in range(BATCH_SIZE)])).to(device)

    loss_sum = 0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Avg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
    for batch_i, (batch_images, batch_actions, batch_subtasks, batch_episode_id, worker_ids, finished_episodes) in pbar:

        print(f"{worker_ids=}")
        print(f"{finished_episodes=}")
        print(f"{worker_ids[finished_episodes]=}")

        # Model

        batch_loss = model_forward_batch(batch_actions,
                                         (batch_images, batch_subtasks),
                                         (batch_images,),
                                         dummy_first,
                                         batch_episode_id,
                                         worker_ids,
                                         original_policy,
                                         policy)

        #
        # for image, action, subtasks, episode_id, is_last in zip(
        #         batch_images, batch_actions, batch_subtasks, batch_unique_id, finished_episodes
        # ):
        #     if is_last:
        #         removed_hidden_state = episode_hidden_states.pop(episode_id)
        #         del removed_hidden_state

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

        if finished_episodes.any():
            # print(f'{finished_episodes.any()=}')
            policy.reset_states(ids=worker_ids[finished_episodes])

        # end = time.time(); print("4.", end - start); start = time.time()

    state_dict = policy.state_dict()
    torch.save(state_dict, out_weights)


if __name__ == "__main__":
    behavioural_cloning_train_teacher_batch(
        data_dir="../basalt_neurips_data/MineRLBasaltMakeWaterfall-v0",
        in_model_original="./data/VPT-models/2x.model",
        in_model="./data/VPT-models/2x.model",
        in_weights_original="./data/VPT-models/foundation-model-2x.weights",
        in_weights="./data/agent_st/foundation-model-tl-tt-2x.weights",
        out_weights="./data/agent_st/foundation-model-tl-bct-2x.weights",
        environment="MineRLBasaltFindCave-v0"
    )






