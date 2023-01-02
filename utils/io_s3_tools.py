from concurrent.futures import as_completed
from typing import List

from tqdm import tqdm

from utils.invalid_files import pp_video, pp_actions, pp_annotations
import boto3
import os

import boto3
import os
from concurrent.futures import ThreadPoolExecutor
import logging


LEVEL = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(LEVEL)
console_handler = logging.StreamHandler()
console_handler.setLevel(LEVEL)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_s3_files(s3, bucket_name, folder: str) -> List:
    objects = []

    # Create a reusable Paginator
    paginator = s3.get_paginator('list_objects')

    # Create a PageIterator from the Paginator, filtered by the specified folder prefix
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder)

    for page in page_iterator:
        objects += page['Contents']

    return objects


def get_s3_files_ends_with(s3, bucket_name, folder: str, ends_with) -> List[str]:
    objects = get_s3_files(s3, bucket_name, folder)

    objects = [obj['Key'] for obj in objects if obj['Key'].endswith(ends_with)]
    return objects


def threaded_download_from_s3(s3, bucket_name, files, local_folder, threads: int = 30, show_progress: bool = False,
                              log_errors: bool = False):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        if show_progress:
            it = tqdm(files)
        # Submit tasks to download the files from S3
        # print(f"{bucket_name=} {local_folder=} {files[0]=}")
        # print(files)
        futures = [executor.submit(s3.download_file, bucket_name, file, os.path.join(local_folder, file)) for file in files]

        # Iterate over the completed tasks
        for future in as_completed(futures):
            if show_progress:
                it.update()
            # Check the status of the task
            if future.exception():
                if log_errors:
                    logger.error(f"Error downloading file: {future.result()}")
                for future_ in tqdm(futures, desc="shutdown"):
                    future_.shutdown(wait=False)
                return False
    return True


if __name__ == "__main__":
    def download_folder_from_s3(bucket_name, s3_prefix, local_folder, file_extensions, threads=30):
        # Create an S3 client
        s3 = boto3.client('s3')

        objects = [j for i in [get_s3_files_ends_with(s3, "basalt-neurips", s3_prefix, ext) for ext in file_extensions]
                   for j in i]
        # Create a ThreadPoolExecutor with a certain number of threads

        threaded_download_from_s3(s3, bucket_name=bucket_name,
                                  files=objects,
                                  local_folder=local_folder,
                                  threads=threads,
                                  show_progress=True,
                                  log_errors=True)


    download_folder_from_s3(bucket_name='basalt-neurips',
                            s3_prefix='basalt_neurips_data/MineRLBasaltMakeWaterfall-v0/',
                            local_folder='../../',
                            file_extensions=[pp_video, pp_actions, pp_annotations],
                            threads=30)


