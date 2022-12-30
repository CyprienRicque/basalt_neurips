import json
import logging
import sys
sys.path.append("../")
from src.lib.action_mapping import CameraHierarchicalMapping


filename = "lovely-persimmon-angora-282ab1fa38e7-20220717-001240_preprocessed.jsonl"
dir = "/media/Data/Documents/Github/basalt_neurips_data/MineRLBasaltMakeWaterfall-v0/"
file_path = dir + filename


if __name__ == "__main__":
    jsons = {}

    with open(file_path, 'r') as f:
        # Load the JSON objects from the file and store them in the dictionary
        for i, line in enumerate(f):
            obj = json.loads(line)
            if obj:  # skip empty JSON objects
                jsons[str(i)] = obj
            # print(obj['buttons'], obj['buttons'][0])
            print(CameraHierarchicalMapping.BUTTONS_IDX_TO_COMBINATION[obj['buttons'][0][0]])



