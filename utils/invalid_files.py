import logging
import os

from src.data_loader import EXT_FORMAT
from utils.download_dataset import is_valid_jsonl, is_valid_mp4, remove_if_exists
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


directory = "./../basalt_neurips_data/MineRLBasaltMakeWaterfall-v0/"

"""
Files architecture:
- filename + .mp4
- filename + .jsonl
- filename + _annotations.jsonl
- filename + _preprocessed.mp4
- filename + _preprocessed.jsonl
- filename + _preprocessed_annotations.jsonl
"""



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


import os


raw_video = ".mp4"
raw_actions = ".jsonl"
raw_annotations = "_annotations.jsonl"
pp_video = "_preprocessed.mp4"
pp_actions = "_preprocessed.jsonl"
pp_annotations = "_preprocessed_annotations.jsonl"

is_pp_annotations = lambda x: x.endswith(pp_annotations)
is_pp_actions = lambda x: x.endswith(pp_actions)
is_pp_video = lambda x: x.endswith(pp_video)

is_raw_annotations = lambda x: x.endswith(raw_annotations) and not is_pp_annotations(x)
is_raw_actions = lambda x: x.endswith(raw_actions) and not is_pp_actions(x)
is_raw_video = lambda x: x.endswith(raw_video) and not is_pp_video(x)

rules = [
    {"remove_all_if_one_is_missing": [raw_video, raw_actions]},
    {"remove_all_if_one_is_missing": [pp_video, pp_actions]},
    {"warning": {"if_present": [raw_video, raw_actions], "warning_if_absent": [pp_video, pp_actions]}},
    {"warning": {"if_present": [raw_annotations], "warning_if_absent": [pp_annotations]}},
]


def check_files(directory):
    # Get a list of all the files in the directory
    files = os.listdir(directory)
    checked = []
    # Iterate through each file in the directory
    for file in files:
        # print(file)
        file = file.replace(EXT_FORMAT, "").replace("_annotations", "")
        # Get the filename without the extension
        filename, file_extension = os.path.splitext(file)
        # print(filename, file_extension)

        if filename in checked:
            continue
        checked.append(filename)

        for rule in rules:
            if "remove_all_if_one_is_missing" in rule:
                delete_all = False
                for extension in rule["remove_all_if_one_is_missing"]:
                    delete_all = delete_all or not os.path.exists(f"{directory}/{filename}{extension}")
                    # if not os.path.exists(f"{directory}/{filename}{extension}"):
                    #     print(f"{directory}/{filename}{extension}", "does not exist")
                if delete_all:
                    for extension in rule["remove_all_if_one_is_missing"]:
                        # remove_if_exists(f"{directory}/{filename}{extension}")
                        logger.info(f"remove {filename}{extension}")

            if "warning" in rule:
                all_present = True
                for extension in rule["warning"]["if_present"]:
                    all_present = all_present or not os.path.exists(f"{directory}/{filename}{extension}")
                if not all_present:
                    continue
                for extension in rule["warning"]["warning_if_absent"]:
                    all_present = all_present or not os.path.exists(f"{directory}/{filename}{extension}")
                if all_present:
                    logger.warning(f"{filename} {rule['warning']['warning_if_absent']} is/are absent while {rule['warning']['if_present']} is/are present")


if __name__ == "__main__":
    check_files("../basalt_neurips_data/MineRLBasaltMakeWaterfall-v0/")
