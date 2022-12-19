import json
import logging


class JSONLManager:
    def __init__(self, file_path, num_jsons):
        self.file_path = file_path
        self.num_jsons = num_jsons
        self.jsons = {}

        # Open the file if it exists, or create it if it doesn't
        try:
            with open(self.file_path, 'r') as f:
                # Load the JSON objects from the file and store them in the dictionary
                for i, line in enumerate(f):
                    obj = json.loads(line)
                    if obj:  # skip empty JSON objects
                        self.jsons[str(i)] = obj
        except FileNotFoundError:
            with open(self.file_path, 'w') as f:
                pass

    def add_json(self, index, obj):
        # Convert the index to a string and add the JSON object to the dictionary
        if index >= self.num_jsons:
            logging.warning(f"object at index {index} will not be dump because greater than limit {self.num_jsons}")
        if index < 0:
            logging.warning(f"object at index {index} will not be dump lower than 0")
        self.jsons[str(index)] = obj

    def reset(self):
        # Clear the dictionary of JSON objects
        self.jsons = {}

    def dump(self):
        # Open the file in write mode
        with open(self.file_path, 'w') as f:
            # Write each JSON object in the dictionary to a separate line in the file
            for i in range(self.num_jsons):
                if str(i) in self.jsons:
                    json.dump(self.jsons[str(i)], f)
                else:
                    json.dump({}, f)
                f.write('\n')


if __name__ == "__main__":
    jm = JSONLManager("coucou.jsonl", 10)
    jm.add_json(10, {"oui": 1})
    jm.add_json(-1, {"oui": 1})
    jm.add_json(0, {"Non": 1})
    jm.dump()
