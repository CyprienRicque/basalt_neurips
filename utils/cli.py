import datetime
import logging
import os
import statistics
import time
import sys
from tqdm import tqdm

sys.path.append('./')

from labeling_gui.label import run_labeler

# CLI for dataset management

# Argument parser
"""
Chose between the 4 main functionalities and then add needed parameters

- download-dataset
    - json-file [required]
    - output-dir [required]
    - num-demos (default=-1)
    - verbose (default=0)
- list-invalid-videos
    - verbose (default=0)
- list-invalid-actions
    - verbose (default=0)
- rm-invalid-duos
    - verbose (default=1)

Example 1:
python cli.py download-dataset --json-file "file.json" output-dir "../data" --verbose=2

Example 2:
python cli.py rm-invalid-duos

"""

import argparse
import json
import os
import sys
sys.path.append("./")

from utils.download_dataset import download_dataset
from utils.invalid_files import list_invalid_videos, remove_duo_when_one_is_invalid, list_invalid_actions

parser = argparse.ArgumentParser()

# Add the main functionality arguments
subparsers = parser.add_subparsers(title="Functionalities", dest="functionality")
download_parser = subparsers.add_parser("download-dataset")
list_invalid_videos_parser = subparsers.add_parser("list-invalid-videos")
list_invalid_actions_parser = subparsers.add_parser("list-invalid-actions")
rm_invalid_duos_parser = subparsers.add_parser("rm-invalid-duos")
total_duration_parser = subparsers.add_parser("total-duration")
delete_annotations_parser = subparsers.add_parser("delete-annotations")
annotate_parser = subparsers.add_parser("annotate")
summary_parser = subparsers.add_parser("summary")

# Add arguments for the "download-dataset" functionality
download_parser.add_argument("--json-file", type=str, required=True, help="Path to the JSON file containing the dataset information")
download_parser.add_argument("--output-dir", type=str, required=True, help="Directory where the dataset will be saved")
download_parser.add_argument("--num-demos", type=int, default=-1, help="Number of demos to download")
download_parser.add_argument("--verbose", type=int, default=0, help="Show verbose output")

# Add arguments for the "list-invalid-videos" functionality
list_invalid_videos_parser.add_argument("--directory", type=str, required=True, help="Directory containing the files")
list_invalid_videos_parser.add_argument("--verbose", type=int, default=1, help="Show verbose output")

# Add arguments for the "list-invalid-actions" functionality
list_invalid_actions_parser.add_argument("--directory", type=str, required=True, help="Directory containing the files")
list_invalid_actions_parser.add_argument("--verbose", type=int, default=1, help="Show verbose output")

# Add arguments for the "rm-invalid-duos" functionality
rm_invalid_duos_parser.add_argument("--directory", type=str, required=True, help="Directory containing the files")
rm_invalid_duos_parser.add_argument("--verbose", type=int, default=1, help="Show verbose output")

# Add arguments for the "total-duration" functionality
total_duration_parser.add_argument("--directory", type=str, required=True, help="Directory containing the files")
total_duration_parser.add_argument("--verbose", type=int, default=1, help="Show verbose output")

# Add arguments for the "total-duration" functionality
delete_annotations_parser.add_argument("--directory", type=str, required=True, help="Directory containing the files")
delete_annotations_parser.add_argument("--verbose", type=int, default=1, help="Show verbose output")
delete_annotations_parser.add_argument("--minutes", type=int, default=-1, help="How recent are the files to be deleted in minutes")

# Add arguments for the "total-duration" functionality
annotate_parser.add_argument("--directory", type=str, required=True, help="Directory containing the files")
annotate_parser.add_argument("--verbose", type=int, default=1, help="Show verbose output")

# Add arguments for the "summary" functionality
summary_parser.add_argument("--directory", type=str, required=True, help="Directory containing the files")
summary_parser.add_argument("--verbose", type=int, default=1, help="Show verbose output")


# Parse the command line arguments
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)
if args.verbose > 1:
    logging.basicConfig(level=logging.DEBUG)

# Handle the "download-dataset" functionality
if args.functionality == "download-dataset":
    download_dataset(args.json_file, args.output_dir, args.num_demos, args.verbose)

if args.functionality == "list-invalid-videos":
    list_invalid_videos(args.directory)

if args.functionality == "list-invalid-actions":
    list_invalid_actions(args.directory)

if args.functionality == "rm-invalid-duos":
    remove_duo_when_one_is_invalid(args.directory)

if args.functionality == "total-duration":
    from moviepy.video.io.VideoFileClip import VideoFileClip

    folder_path = os.path.join(args.directory)

    def compute_video_duration(folder_path):
        durations = []
        total_duration = 0

        pbar = tqdm(enumerate(os.listdir(folder_path)), unit='files')
        for i, file in pbar:
            if file.endswith('.mp4'):
                video = VideoFileClip(os.path.join(folder_path, file))
                duration = video.duration
                durations.append(duration)
                total_duration += duration

                if i % 200 == 0:
                    mean_duration = statistics.mean(durations)
                    median_duration = statistics.median(durations)
                    pbar.set_description(f"total: {str(datetime.timedelta(seconds=total_duration)).split('.')[0]}. mean: {str(datetime.timedelta(seconds=mean_duration)).split('.')[0]}. med: {str(datetime.timedelta(seconds=median_duration)).split('.')[0]}")

        mean_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)

        return mean_duration, median_duration, total_duration

    mean_duration, median_duration, total_duration = compute_video_duration(folder_path)

    print(f"Mean duration: {str(datetime.timedelta(seconds=mean_duration)).split('.')[0]}")
    print(f"Median duration: {str(datetime.timedelta(seconds=median_duration)).split('.')[0]}")
    print(f"Total duration: {str(datetime.timedelta(seconds=total_duration)).split('.')[0]}")


def delete_recent_annotations(folder, nb_minutes):
    if nb_minutes == 0:
        raise ValueError("nb_minutes cannot be 0")

    # Get the current time in seconds
    current_time = time.time()

    # Iterate through all files in the folder
    for file in os.listdir(folder):
        # Check if the file ends with _annotations.jsonl
        if file.endswith("_annotations.jsonl"):
            # Get the file's creation time in seconds
            file_creation_time = os.path.getctime(os.path.join(folder, file))

            # Calculate the difference between the current time and the file's creation time
            time_difference = current_time - file_creation_time

            # Convert the difference to minutes
            time_difference_in_minutes = time_difference / 60

            # Check if the difference is greater than the specified number of minutes
            if nb_minutes == -1 or time_difference_in_minutes < nb_minutes:
                # Delete the file
                os.remove(os.path.join(folder, file))


if args.functionality == "delete-annotations":
    delete_recent_annotations(f"{args.directory}/", int(args.minutes))


if args.functionality == "annotate":
    run_labeler(f"{args.directory}/")


if args.functionality == "summary":
    def summarize_files(directory):
        # Initialize counters for each type of file
        mp4_count = 0
        jsonl_count = 0
        preprocessed_mp4_count = 0
        preprocessed_jsonl_count = 0
        annotations_jsonl_count = 0
        preprocessed_annotations_jsonl_count = 0
        missing_preprocessed_mp4_filenames = []

        # Iterate through the files in the directory
        for filename in os.listdir(directory):
            # Increment the appropriate counter for each file type
            if filename.endswith('_preprocessed.mp4'):
                preprocessed_mp4_count += 1
            elif filename.endswith('_preprocessed_annotations.jsonl'):
                preprocessed_annotations_jsonl_count += 1
            elif filename.endswith('_annotations.jsonl'):
                annotations_jsonl_count += 1
            elif filename.endswith('.mp4'):
                mp4_count += 1
            elif filename.endswith('_preprocessed.jsonl'):
                preprocessed_jsonl_count += 1
            elif filename.endswith('.jsonl'):
                jsonl_count += 1
            if filename.endswith('.mp4') and not filename.endswith('_preprocessed.mp4') and \
                not os.path.exists(directory + "/" + filename.replace(".mp4", "_preprocessed.mp4")):
                missing_preprocessed_mp4_filenames.append(directory + "/" + filename)

        # Print the summary
        print(f'Number of .mp4 files: {mp4_count}')
        print(f'Number of .jsonl files: {jsonl_count}')
        print(f'Number of _annotations.jsonl files: {annotations_jsonl_count}')
        print(f'Number of _preprocessed.mp4 files: {preprocessed_mp4_count}')
        print(f'Number of _preprocessed.jsonl files: {preprocessed_jsonl_count}')
        print(f'Number of _preprocessed_annotations.jsonl files: {preprocessed_annotations_jsonl_count}')
        # If the list of filenames that don't have their related _preprocessed.mp4 files is not empty, print the list
        if missing_preprocessed_mp4_filenames:
            print('Filenames missing _preprocessed.mp4 files:')
            for filename in missing_preprocessed_mp4_filenames:
                print(filename)

    summarize_files(args.directory)

