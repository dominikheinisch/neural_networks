from neural_networks.neuron.simulations import simulation
from neural_networks.neuron.helpers import input_checker
from neural_networks.neuron.io import read, save

import os.path
from sys import argv

SOURCE_FOLDER = os.path.join('data', '{}')


def set_obligatory_params():
    learning_param = 0.2
    scope = (-0.5, 0.5)
    # scope = (-1.0, 1.0)
    # scope = (-0.1, 0.1)
    return {'learning_param': learning_param, 'scope': scope, 'threshold': 1.0}


def set_optional_params(params):
    bias = 0
    params['bias'] = bias
    learning_pairs = {(0, 0): 0, (1, 0): 0, (1, 1): 1, (0, 1): 0}
    params['learning_pairs'] = learning_pairs
    return params


def watch_simulation(params, times=100):
    if len(argv) == 1:
        neuron_class = 'Adaline'
        # neuron_class = 'BasicPerceptron'
        # activation_func = 'binary'
        activation_func = 'bipolar'
    else:
        input_checker.check_input()
        neuron_class, activation_func = argv[2], argv[3]
    time = simulation.run(neuron_class=neuron_class, activation_func=activation_func, params=params,
                          times=times, is_to_plot = True, is_to_print=True)
    # print(time)


def calc_and_save_multiple_simuation():
    filename_pattern = SOURCE_FOLDER.format('learning_param_0.01_0.5{0}{1}{2}.json')
    # filename_pattern = SOURCE_FOLDER.format('scope_(-1, 1)_(-0.2, 0.2){0}{1}{2}.json')
    filename_data = filename_pattern.format('', '', '')
    neuron_name = 'Adaline'
    activation_func = 'bipolar'
    filename_dest = filename_pattern.format('_result_', neuron_name + '_', activation_func)
    save(filename_dest, (simulation.run_many_params(neuron_name, activation_func, times=1000,
                                                    list_params=read(filename_data))))

def run_given_data():
    params = set_obligatory_params()
    # params = set_optional_params(params)
    watch_simulation(params)

def run_one():
    # watch_simulation(read(SOURCE_FOLDER.format('default_input_threshold_0.5.json')), times=2)
    # watch_simulation(read(SOURCE_FOLDER.format('default_input_threshold_1.0.json')), times=2)
    # watch_simulation(read(SOURCE_FOLDER.format('default_input_threshold_1.5.json')), times=20)
    watch_simulation(read(SOURCE_FOLDER.format('default_input_threshold_2.0.json')), times=20)


if __name__ == "__main__":
    print('NAKLEJKA LEGITYMACJA')
    # run_one()
    # run_given_data()
    calc_and_save_multiple_simuation()
    # print(simulation.run_avg_epochs('BasicPerceptron', 'bipolar', params, times=1000))
    # print(simulation.run_many_params('BasicPerceptron', 'bipolar', times=1000, list_params=read('data/learning_param_0.01_0.5.json')))



