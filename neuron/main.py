from neural_networks.neuron.simulations import simulation
from neural_networks.neuron.helpers import input_checker
from neural_networks.neuron.io import read, save

from sys import argv


def set_obligatory_params():
    # learning_param = 0.05
    learning_param = 0.01
    scope = (-1.0, 1.0)
    # scope = (-0.1, 0.1)
    return {'learning_param': learning_param, 'scope': scope, 'threshold': 1.5}


def set_optional_params(params):
    bias = 0
    params['bias'] = bias
    learning_pairs = {(0, 0): 0, (1, 0): 0, (1, 1): 1, (0, 1): 0}
    params['learning_pairs'] = learning_pairs
    return params


def watch_simulation(params, times=100):
    if len(argv) == 1:
        # neuron_class = 'Adaline'
        neuron_class = 'BasicPerceptron'
        # activation_func = 'binary'
        activation_func = 'bipolar'
    else:
        input_checker.check_input()
        neuron_class, activation_func = argv[2], argv[3]
    time = simulation.run(neuron_class=neuron_class, activation_func=activation_func, params=params,
                          times=times, is_to_plot = True, is_to_print=True)
    # print(time)


def calc_and_save_multiple_simuation():
    # filename_pattern = 'data/learning_param_0.01_0.5{0}{1}{2}.json'
    filename_pattern = 'data/scope_(-1, 1)_(-0.2, 0.2){0}{1}{2}.json'
    filename_data = filename_pattern.format('', '', '')
    neuron_name = 'BasicPerceptron'
    activation_func = 'binary'
    filename_dest = filename_pattern.format('_result_', neuron_name + '_', activation_func)
    save(filename_dest, (simulation.run_many_params(neuron_name, activation_func, times=1000,
                                                    list_params=read(filename_data))))


if __name__ == "__main__":
    # params = set_obligatory_params()
    # params = set_optional_params(params)

    watch_simulation(read('data/default_input_1.json'), times=2)
    watch_simulation(read('data/default_input_2.json'), times=2)

    # print(simulation.run_avg_epochs('BasicPerceptron', 'bipolar', params, times=1000))

    # print(simulation.run_many_params('BasicPerceptron', 'bipolar', times=1000, list_params=read('data/learning_param_0.01_0.5.json')))
    # calc_and_save_multiple_simuation()


