# Code for loading OpenAI MineRL VPT datasets
# (NOTE: Not the original code!)
import json
import glob
import logging
import os
import random
from multiprocessing import Process, Queue, Event

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.agent import MineRLAgent
from src.lib.tree_util import tree_map
from src.original_agent import resize_image, AGENT_RESOLUTION

import logging

from src.utils import count_lines

EXT_FORMAT = "_preprocessed"
LEVEL = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(LEVEL)
console_handler = logging.StreamHandler()
console_handler.setLevel(LEVEL)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

QUEUE_TIMEOUT = 60

CURSOR_FILE = os.path.join(
    os.path.dirname(__file__), "../cursors", "mouse_cursor_white_16x16.png"
)


def data_loader_worker(tasks_queue, output_queue, quit_workers_event, apply_bgr2rgb):
    """
    Worker for the data loader.
    """

    while True:
        task = tasks_queue.get()  # For each file
        if task is None:  # Nothing left to do
            break
        unique_id, video_path, env_json_path, st_json_path = task
        video = cv2.VideoCapture(video_path)
        # Note: In some recordings, the game seems to start with attack always down from the beginning, which
        #       is stuck down until player actually presses attack
        # Scrollwheel is allowed way to change items, but this is not captured by the recorder.
        # Work around this by keeping track of selected hotbar item and updating "hotbar.#" actions when hotbar selection changes.

        with open(env_json_path) as env_json_file:
            env_json_lines = env_json_file.readlines()
            env_json_data = "[" + ",".join(env_json_lines) + "]"
            env_json_data = json.loads(env_json_data)

        with open(st_json_path) as st_json_file:
            st_json_lines = st_json_file.readlines()
            st_json_data = "[" + ",".join(st_json_lines) + "]"
            st_json_data = json.loads(st_json_data)

        for i in range(len(env_json_data)):  # For each frame
            if quit_workers_event.is_set():
                break
            step_env_data = env_json_data[i]
            step_st_data = st_json_data[i]

            # Read subtasks
            subtasks = torch.tensor(step_st_data['one_hot'], dtype=torch.uint8)
            action = {key: torch.tensor(value) for key, value in step_env_data.items()}

            # Read frame even if this is null, so we progress forward
            ret, frame = video.read()
            if ret:
                # Skip null actions as done in the VPT paper
                # NOTE: in VPT paper, this was checked _after_ transforming into agent's action-space.
                #       We do this here as well to reduce amount of data sent over.
                if apply_bgr2rgb:
                    cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = torch.from_numpy(np.asarray(np.clip(frame, 0, 255), dtype=np.uint8))
                # print("is last?", True if i == len(env_json_data) - 1 else False)
                output_queue.put((
                    unique_id,
                    frame.unsqueeze(0),
                    {"buttons": action['buttons'], "camera": action['camera']},
                    subtasks.unsqueeze(0),
                    True if i == len(env_json_data) - 1 else False
                ), timeout=QUEUE_TIMEOUT)

            else:
                logging.warning(f"Could not read frame from video {video_path}")

        video.release()
        # Signal that this task is done
        # Yes we are using "None"s to tell when worker is done and when individual work-items are done...
        # output_queue.put((unique_id, None, None, None), timeout=QUEUE_TIMEOUT)
        if quit_workers_event.is_set():
            break
    # Tell that we ended
    output_queue.put(None)


class DataLoader:
    """
    Generator class for loading batches from a dataset

    This only returns a single step at a time per worker; no sub-sequences.
    Idea is that you keep track of the model's hidden state and feed that in,
    along with one sample at a time.

    + Simpler loader code
    + Supports lower end hardware
    - Not very efficient (could be faster)
    - No support for sub-sequences
    - Loads up individual files as trajectory files (i.e. if a trajectory is split into multiple files,
      this code will load it up as a separate item).
    """

    def __init__(
            self,
            dataset_dir: str,
            n_workers: int = 8,
            batch_size: int = 8,
            n_epochs: int = 1,
            max_queue_size: int = 24,
            dataset_max_size: int = -1,
            shuffle: bool = True,
            apply_bgr2rgb: bool = True,
            exclude=lambda id_: False,
            device: str = "cuda",
    ):
        """

        :param dataset_dir:
        :param n_workers:
        :param batch_size:
        :param n_epochs:
        :param max_queue_size:
        :param dataset_max_size: maximum number of trajectories to load from the dataset. -1 means no limit.
        """
        self.device = device
        assert (
                n_workers >= batch_size
        ), "Number of workers must be equal or greater than batch size"
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.apply_bgr2rgb = apply_bgr2rgb

        if shuffle is False:
            logging.warning("Shuffle is set to false")

        unique_ids = glob.glob(os.path.join(dataset_dir, f"*{EXT_FORMAT}.mp4"))
        unique_ids = [id_ for id_ in unique_ids if (not exclude(id_))]
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = unique_ids[:dataset_max_size] if dataset_max_size != -1 else unique_ids

        # Create tuples of (video_path, json_path) for each unique_id
        demonstration_tuples = []
        for unique_id in self.unique_ids:
            video_path = os.path.abspath(os.path.join(dataset_dir, f"{unique_id}.mp4"))
            env_json_path = os.path.abspath(os.path.join(dataset_dir, f"{unique_id}.jsonl"))
            st_json_path = os.path.abspath(os.path.join(dataset_dir, f"{unique_id}_annotations.jsonl"))

            demonstration_tuples.append((unique_id, video_path, env_json_path, st_json_path))

        assert n_workers <= len(
            demonstration_tuples
        ), f"n_workers should be lower or equal than number of demonstrations {len(demonstration_tuples)}"

        # Repeat dataset for n_epochs times, shuffling the order of files for each epoch
        self.demonstration_tuples = []
        for i in range(n_epochs):
            if shuffle:
                random.shuffle(demonstration_tuples)
            self.demonstration_tuples += demonstration_tuples

        self.task_queue = Queue()
        self.n_steps_processed = 0
        for trajectory_id, task in enumerate(self.demonstration_tuples):
            self.task_queue.put(task)

        for _ in range(n_workers):  # So that each worker detects when queue is empty
            self.task_queue.put(None)

        self.output_queues = [Queue(maxsize=max_queue_size) for _ in range(n_workers)]
        self.quit_workers_event = Event()
        self.processes = [
            Process(
                target=data_loader_worker,
                args=(
                    self.task_queue,
                    output_queue,
                    self.quit_workers_event,
                    self.apply_bgr2rgb,
                ),
                daemon=True,
            )
            for output_queue in self.output_queues
        ]
        for process in self.processes:
            process.start()

    def __iter__(self):
        return self

    def __next__(self):
        batch_frames, batch_actions, batch_subtasks, batch_unique_id, finished_episodes, worker_ids = [], [], [], [], [], []

        for i in range(self.batch_size):
            worker_id = self.n_steps_processed % self.n_workers
            work_item = self.output_queues[worker_id].get(timeout=QUEUE_TIMEOUT)
            if work_item is None:
                # Stop iteration when first worker runs out of work to do. Yes, this has a chance of cutting out a lot
                # of the work, but this ensures batches will remain diverse, instead of having bad ones in the end where
                # potentially one worker outputs all samples to the same batch.
                raise StopIteration("First worker ran out of work")

            trajectory_id, frame, action, subtasks, is_last = work_item

            # batch_frames.append(frame.to(self.device))
            # batch_actions.append(
            #     {"buttons": action['buttons'].to(self.device), "camera": action['camera'].to(self.device)})
            # batch_subtasks.append(subtasks.to(self.device))
            # print(f"{worker_id=}")
            batch_frames.append(frame)
            batch_actions.append({"buttons": action['buttons'], "camera": action['camera']})
            batch_subtasks.append(subtasks)
            batch_unique_id.append(trajectory_id)
            worker_ids.append(worker_id)

            finished_episodes.append(is_last)
            self.n_steps_processed += 1
        #
        # print(f"{batch_subtasks=}")
        # # print(batch_frames)
        # print(
        #     # torch.tensor(batch_frames, device=self.device),
        #     # batch_actions,
        #     torch.tensor(batch_subtasks, device=self.device),
        #     # torch.tensor(batch_unique_id, device=self.device),
        #     # torch.tensor(worker_ids, device=self.device),
        #     # finished_episodes
        #       )

        return torch.cat(batch_frames).to(device=self.device, non_blocking=True), \
            batch_actions, \
            torch.cat(batch_subtasks).to(device=self.device, non_blocking=True), \
            batch_unique_id, \
            torch.tensor(worker_ids, device=self.device), \
            torch.tensor(finished_episodes, device=self.device)

    def __len__(self):
        total_lines = count_lines([env_json_path for _, _, env_json_path, _ in self.demonstration_tuples])
        return int(total_lines / self.batch_size) * self.n_epochs

    def __del__(self):
        self.quit_workers_event.set()
        for process in self.processes:
            process.terminate()
            process.join()

# ======================================================================================================================
# class MinecraftDataset(Dataset):
#     def __init__(self, directory):
#         self.directory = directory
#
#         unique_ids = glob.glob(os.path.join(directory, "*_preprocessed.mp4"))
#         unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
#
#         # Create tuples of (video_path, json_path, json_path) for each unique_id
#         demonstration_tuples = []
#
#         for unique_id in unique_ids:
#             video_path = os.path.abspath(os.path.join(directory, unique_id + ".mp4"))
#             env_json_path = os.path.abspath(os.path.join(directory, unique_id + ".jsonl"))
#             st_json_path = os.path.abspath(os.path.join(directory, unique_id + "_annotations.jsonl"))
#             demonstration_tuples.append((unique_id, video_path, env_json_path, st_json_path))
#
#         self.demonstration_tuples = demonstration_tuples
#
#
#     def __len__(self):
#         total_lines = count_lines([env_json_path for _, _, env_json_path, _ in self.demonstration_tuples])
#         return total_lines
#
#     def __getitem__(self, idx):
#         while True:
#             task = tasks_queue.get()  # For each file
#             if task is None:  # Nothing left to do
#                 break
#             trajectory_id, video_path, env_json_path, st_json_path = task
#             video = cv2.VideoCapture(video_path)
#             # Note: In some recordings, the game seems to start with attack always down from the beginning, which
#             #       is stuck down until player actually presses attack
#             attack_is_stuck = False
#             # Scrollwheel is allowed way to change items, but this is not captured by the recorder.
#             # Work around this by keeping track of selected hotbar item and updating "hotbar.#" actions when hotbar selection changes.
#             last_hotbar = 0
#
#             with open(env_json_path) as env_json_file:
#                 env_json_lines = env_json_file.readlines()
#                 env_json_data = "[" + ",".join(env_json_lines) + "]"
#                 env_json_data = json.loads(env_json_data)
#
#             try:
#                 with open(st_json_path) as st_json_file:
#                     st_json_lines = st_json_file.readlines()
#                     st_json_data = "[" + ",".join(st_json_lines) + "]"
#                     st_json_data = json.loads(st_json_data)
#             except FileNotFoundError:  # Security for when this is not annotated.
#                 st_json_data = [{} for _ in range(len(env_json_data))]
#
#             for i in range(len(env_json_data)):  # For each frame
#                 if quit_workers_event.is_set():
#                     break
#                 step_env_data = env_json_data[i]
#                 try:
#                     step_st_data = st_json_data[i]
#                 except:
#                     print(f"index error {len(st_json_data)=} {st_json_path=} {i=}", flush=True)
#                     logger.error(f"index error {len(st_json_data)=} {i=}")
#                     raise Exception
#                     # exit(1)
#                 # print(f"process frame {env_json_path=}")
#
#                 attack_is_stuck = process_frame(
#                     attack_is_stuck,
#                     cursor_alpha,
#                     cursor_image,
#                     i,
#                     step_env_data,
#                     step_st_data,
#                     last_hotbar,
#                     output_queue,
#                     trajectory_id,
#                     video,
#                     video_path,
#                     apply_bgr2rgb,
#                     env_json_path,
#                 )
#             video.release()
#             # Signal that this task is done
#             # Yes we are using "None"s to tell when worker is done and when individual work-items are done...
#             output_queue.put((trajectory_id, None, None, video_path.replace(".mp4", "")), timeout=QUEUE_TIMEOUT)
#             if quit_workers_event.is_set():
#                 break
#         # Tell that we ended
#         output_queue.put(None)
#
#         return image, label
#
# if __name__ == "__main__":
