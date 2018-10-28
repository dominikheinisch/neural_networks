import numpy as np


def preapre(params):
    add_default_params(params)
    adjust_params(params)


def add_default_params(params):
    if 'learning_pairs' not in params.keys():
        params['learning_pairs'] = {(0, 0): 0, (1, 0): 0, (1, 1): 1, (0, 1): 0}
    if 'activation_func' not in params.keys():
        params['activation_func'] = 'binary'
    if 'bias' not in params.keys():
        params['bias'] = 1


def transform_number(x):
    return x + round(x) - 1


def input_to_bipolar(input_pair):
    return transform_number(input_pair[0]), transform_number(input_pair[1])


def output_to_bipolar(output):
    return transform_number(output)


def adjust_params(params):
    if params['activation_func'] == 'binary':
        params['invalid_output'] = 0
    elif params['activation_func'] == 'bipolar':
        params['invalid_output'] = -1
        params['learning_pairs'] = {input_to_bipolar(pair): output_to_bipolar(params['learning_pairs'][pair])
                                    for pair in params['learning_pairs'].keys()}
    params['learning_pairs'] = transform_learning_pairs(params['learning_pairs'], params['bias'])


def transform_learning_pairs(params, bias):
    return {x: (np.array([bias, *x]), params[x]) for x in params}
