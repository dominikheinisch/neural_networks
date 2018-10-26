from neural_networks.neuron.simulations import simulation
from neural_networks.neuron.helpers import input_checker

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


def watch_simulation(params, times=1000):
    if len(argv) == 1:
        # neuron_class, activation_func = 'Adaline', 'binary'
        # neuron_class, activation_func = 'Adaline', 'bipolar'
        # neuron_class, activation_func = 'BasicPerceptron', 'binary'
        neuron_class, activation_func = 'BasicPerceptron', 'bipolar'
    else:
        input_checker.check_input()
        neuron_class, activation_func = argv[2], argv[3]
    time = simulation.run(neuron_class=neuron_class, activation_func=activation_func, params=params,
                          times=times, is_to_plot = True, is_to_print=True)
    print(time)


if __name__ == "__main__":
    params = set_obligatory_params()
    # params = set_optional_params(params)

    # watch_simulation(params)
    print(simulation.run_many_to_get_avg_epochs('BasicPerceptron', 'bipolar', params))



