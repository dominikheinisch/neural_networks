from neural_networks.neuron.io import save

import copy


def generate_data():
    learning_param = 0.05
    scope = (-1.0, 1.0)
    threshold = 1.0
    params = {'learning_param': learning_param, 'scope': scope, 'threshold': threshold}
    save('../data/f1.json', params)


def create_params(params, param_name, value):
    cpy = copy.deepcopy(params)
    cpy[param_name] = value
    return cpy


def generate_many_data(params, param_name, modified_param):
    accumulator = [(param_name, modified_param)]
    accumulator += [create_params(params, param_name, param) for param in modified_param]
    modified_param_str = '_'.join([str(p) for p in (modified_param[0], modified_param[-1])])
    save('../data/{0}_{1}.json'.format(param_name, modified_param_str), accumulator)


if __name__ == "__main__":
    # generate_data()
    # generate_many_data({'scope': (-0.5, 0.5), 'threshold': 1.0}, 'learning_param',
    #                    [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    generate_many_data({'learning_param': 0.1, 'threshold': 1.0}, 'scope',
                       [(-1, 1), (-0.8, 0.8), (-0.5, 0.5), (-0.2, 0.2)])
    # generate_many_data({'scope': (-0.5, 0.5), 'learning_param': 0.05}, 'threshold',
    #                    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
