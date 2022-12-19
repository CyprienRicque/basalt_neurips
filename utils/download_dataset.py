# A script to download OpenAI contractor data or BASALT datasets
# using the shared .json files (index file).
#
# Json files are in format:
# {"basedir": <prefix>, "relpaths": [<relpath>, ...]}
#
# The script will download all files in the relpaths list,
# or maximum of set number of demonstrations,
# to a specified directory.
#

import argparse
import concurrent
import json
import logging
import sys
import time
import urllib.request
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import cv2
import requests
from tqdm import tqdm


CHUNK_SIZE = 1024
MAX_RETRIES = 1
THREADS = 2**7
MAX_TENTATIVES = 3

formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser(description="Download OpenAI contractor datasets")
parser.add_argument("--json-file", type=str, required=True, help="Path to the index .json file")
parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")
parser.add_argument("--num-demos", type=int, default=None, help="Maximum number of demonstrations to download")
parser.add_argument("--verbose", type=int, default=0, help="Set to 1 to print info. 2 to print debug. 0 for default")


def is_valid_mp4(video_path):
    if not os.path.exists(video_path):
        logging.warning(f"file {video_path} does not exist")
        return False
    logging.debug(f"cv2.VideoCapture({video_path})")
    video = cv2.VideoCapture(video_path)
    logging.debug("_________________")

    ret, frame = video.read()
    if not ret:
        logging.info(f"Could not read frame from video {video_path}")
        return False

    video.release()
    return True


def is_valid_jsonl(filepath) -> bool:
    with open(filepath, 'r') as f:
        for line in f:
            try:
                json.loads(line)
            except ValueError:
                return False
    return True


def download_by_chunks(url, path) -> None:
    retries = 0

    # Download the video in chunks until it is fully downloaded
    with open(path, "wb") as f:
        while True:
            # Download a chunk of the video
            try:
                response = requests.get(url, stream=True)

                # If the response is successful, write the chunk to the file
                if response.status_code == 200:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        f.write(chunk)

                    # If the entire video was downloaded, break out of the loop
                    if int(response.headers["Content-Length"]) == f.tell():
                        break

                # If the response was not successful, raise an exception
                else:
                    raise Exception("Failed to download video {}".format(path))

            # If there was an error downloading the chunk, retry if possible
            except:
                if retries < MAX_RETRIES:
                    retries += 1
                    continue
                else:
                    raise


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def download_file(output_dir, filename, url, out_path) -> bool:
    if not os.path.exists(os.path.join(output_dir, filename)):
        try:
            logging.info(f"Downloading {url} to {out_path}")
            download_by_chunks(url, out_path)
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}. Cleaning up {out_path}")
            remove_if_exists(out_path)
            return False
    else:
        logging.info(f"File {filename} already exists in {output_dir}, skipping")
    return True


def download_video_actions(relpath, output_dir, base_dir, tentatives=0):
    download_is_success: bool = True

    logging.info(f"download_file(): {relpath}")

    # video file metadata
    video_url = base_dir + relpath
    video_filename = os.path.basename(relpath)
    video_out_path = os.path.join(output_dir, video_filename)

    # jsonl file metadata
    actions_url = video_url.replace(".mp4", ".jsonl")
    actions_filename = video_filename.replace(".mp4", ".jsonl")
    actions_out_path = os.path.join(output_dir, actions_filename)

    # Download the video file
    download_file(output_dir, video_filename, video_url, video_out_path)

    # Also download corresponding .jsonl file
    download_file(output_dir, actions_filename, actions_url, actions_out_path)

    # Check if the .mp4 file is valid
    if not is_valid_mp4(video_out_path):
        logging.info(f"Invalid mp4 file {video_out_path}, rm & retry. Remaining tentatives: {MAX_TENTATIVES - tentatives}")
        if tentatives < MAX_TENTATIVES:
            remove_if_exists(video_out_path)
            return download_video_actions(relpath, output_dir, base_dir, tentatives + 1)
        else:
            logging.warning(f"file {video_url} is not valid")
            download_is_success = False

    # Check if the .jsonl file is valid
    if not is_valid_jsonl(actions_out_path):
        logging.info(f"Invalid jsonl file {actions_out_path}, rm & retry. Remaining tentatives: {MAX_TENTATIVES - tentatives}")
        if tentatives < MAX_TENTATIVES:
            remove_if_exists(actions_out_path)
            return download_video_actions(relpath, output_dir, base_dir, tentatives + 1)
        else:
            logging.warning(f"file {actions_url} is not valid")
            download_is_success = False

    return download_is_success


def download_dataset(json_file, output_dir, num_demos, verbose):
    with open(json_file, "r") as f:
        data = f.read()
    data = eval(data)
    base_dir = data["basedir"]
    rel_paths = data["relpaths"]

    if verbose:
        logging.basicConfig(level=logging.INFO)
    if verbose > 1:
        logging.basicConfig(level=logging.DEBUG)

    logging.info(f"Total number of links: {len(rel_paths)}. Using {num_demos or len(rel_paths)}")

    if num_demos is not None:
        rel_paths = rel_paths[:num_demos]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.debug(f"{rel_paths=}")
    logging.debug(f"{output_dir=}")
    logging.debug(f"{base_dir=}")

    with tqdm(total=len(rel_paths)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = {executor.submit(download_video_actions,
                                       relpath,
                                       output_dir=output_dir,
                                       base_dir=base_dir): relpath for relpath in rel_paths}
            results = {}
            failed = []
            for future in concurrent.futures.as_completed(futures):
                arg = futures[future]
                results[arg] = future.result()
                if results[arg] is False:
                    failed.append(arg)
                pbar.update(1)

    logging.debug(f"{results=}")
    for i in failed:
        logging.warning(f"Failed to download {i}")


if __name__ == "__main__":
    args = parser.parse_args()
    download_dataset(args.json_file, args.output_dir, args.num_demos, args.verbose)
