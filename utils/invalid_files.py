import logging
import os

from utils.download_dataset import is_valid_jsonl, is_valid_mp4, remove_if_exists

directory = "./../basalt_neurips_data/MineRLBasaltMakeWaterfall-v0/"


def invalid_actions(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            if not is_valid_jsonl(filepath):
                yield filepath


def invalid_videos(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            filepath = os.path.join(directory, filename)
            if not is_valid_mp4(filepath):
                yield filepath


def list_invalid_videos(directory):
    for filepath in invalid_videos(directory):
        print(filepath)


def list_invalid_actions(directory):
    for filepath in invalid_actions(directory):
        print(filepath)


def get_invalid_videos(directory):
    return [path for path in invalid_videos(directory)]


def get_invalid_actions(directory):
    return [path for path in invalid_actions(directory)]


def remove_duo_when_one_is_invalid(directory):
    invalid_actions_list = get_invalid_actions(directory)
    invalid_videos_list = get_invalid_videos(directory)

    for action_path in invalid_actions_list:
        remove_if_exists(action_path)
        logging.info(f"{action_path} removed")

        video_path = action_path.replace(".jsonl", ".mp4")
        remove_if_exists(video_path)
        logging.info(f"{video_path} removed")

    for video_path in invalid_videos_list:
        remove_if_exists(video_path)
        logging.info(f"{video_path} removed")

        action_path = video_path.replace(".mp4", ".jsonl")
        remove_if_exists(action_path)
        logging.info(f"{action_path} removed")
