import json


def read_many_params(file):
    with open(file, 'r') as f:
        loaded = json.load(f)
        return loaded


if __name__ == "__main__":
    print(read_many_params('../data/learning_param_0.01_0.5.json'))