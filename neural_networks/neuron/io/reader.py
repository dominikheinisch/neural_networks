import json


def read_many_params(file):
    with open(file, 'r') as f:
        loaded = json.loads(f)
        print(loaded)


if __name__ == "__main__":
    read_many_params('../data/learning_param_0.01_0.05_0.1_0.2_0.3_0.4_0.5.json')