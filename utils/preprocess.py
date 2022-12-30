import datetime
import logging
import os
import statistics
import time
import sys
from tqdm import tqdm
import cv2

from src.agent import MineRLAgent, AGENT_RESOLUTION
from src.data_loader import DataLoader, EXT_FORMAT
from src.lib.actions import Buttons
import numpy as np

import logging

LEVEL = logging.INFO
N_WORKERS = 8
BATCH_SIZE = 4
EPOCHS = 1

logger = logging.getLogger(__name__)
logger.setLevel(LEVEL)
console_handler = logging.StreamHandler()
console_handler.setLevel(LEVEL)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

sys.path.append('../')

filtered_images = []
filtered_actions = []
filtered_subtasks = []
filtered_episode_ids = []


def export_video(output_filename, filtered_images):
    # Set the codec for the MP4 file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Set the output file name and the frame size
    frame_size = (AGENT_RESOLUTION[1], AGENT_RESOLUTION[0])

    # Create the VideoWriter object
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, frame_size)

    # Loop through the filtered frames and write them to the MP4 file
    for image in filtered_images:
        out.write(image)

    # Release the VideoWriter object
    out.release()


def export_jsonl(output_filename, jsons):
    import json

    # Open the output file in write mode
    with open(output_filename, "w") as f:
        # Loop through the filtered actions and write them to the JSONL file
        for data in jsons:
            # Write the dictionary to the JSONL file as a JSON string
            f.write(json.dumps(data) + "\n")


def rm_noop(data_dir):
    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
        dataset_max_size=-1,
        apply_bgr2rgb=False,
        exclude=lambda id_: (os.path.exists(id_.replace(".mp4", EXT_FORMAT + ".mp4")) and
                             os.path.exists(id_.replace(".mp4", EXT_FORMAT + ".jsonl")) and
                             os.path.exists(id_.replace(".mp4", EXT_FORMAT + "_annotations.jsonl"))) or
                            "hazy-thistle-chipmunk-3709e095cc5e-20220718-201514" in id_ or
                            "lovely-persimmon-angora-7cddf2a5f034-20220715-221133" in id_
    )

    agent = MineRLAgent(
        None,
        device='cpu'
    )

    # Initialize an empty dictionary to store the filtered data
    filtered_data = {}

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_i, (batch_images, batch_actions, batch_subtasks, batch_episode_id) in pbar:
        for image, action, subtasks, episode_id in \
                zip(batch_images, batch_actions, batch_subtasks, batch_episode_id):

            # Initialize the filtered data for this episode ID if it does not exist
            if episode_id not in filtered_data:
                filtered_data[episode_id] = {"images": [], "actions": [], "subtasks": []}

            if image is None and action is None:
                # End of file for this episode ID
                # logger.info(f"End of file {episode_id=}: {subtasks.split('/')[-1]}")
                # Export the video for this episode ID (subtasks value stores file name when the file is ended.)
                export_video(f"{subtasks}{EXT_FORMAT}.mp4", filtered_data[episode_id]["images"])
                export_jsonl(f"{subtasks}{EXT_FORMAT}.jsonl", filtered_data[episode_id]["actions"])
                export_jsonl(f"{subtasks}{EXT_FORMAT}_annotations.jsonl", filtered_data[episode_id]["subtasks"])
                # Remove the filtered data for this episode ID
                del filtered_data[episode_id]
                continue

            agent_action = agent._env_action_to_agent(action, check_if_null=True, to_torch=False)
            if agent_action is None:
                continue

            agent_action = {item: value.tolist() for item, value in agent_action.items()}
            subtasks = {item: value.tolist() for item, value in subtasks.items()}
            # print(f"{subtasks=}")

            # Append the filtered data to the corresponding lists in the dictionary
            filtered_data[episode_id]["images"].append(image)
            filtered_data[episode_id]["actions"].append(agent_action)
            filtered_data[episode_id]["subtasks"].append(subtasks)


if __name__ == '__main__':
    rm_noop("../basalt_neurips_data/MineRLBasaltMakeWaterfall-v0")
