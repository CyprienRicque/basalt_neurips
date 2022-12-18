import os

from utils.download_dataset import is_valid_jsonl, is_valid_mp4

directory = "./../basalt_neurips_data/BuildWaterFall/"


def list_invalid_actions(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            if not is_valid_jsonl(filepath):
                print(filepath)


def list_invalid_videos(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            filepath = os.path.join(directory, filename)
            if not is_valid_mp4(filepath):
                print(filepath)
