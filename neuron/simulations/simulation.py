from neural_networks.neuron.helpers import parameter_preparer
from neural_networks.neuron.neurons import Adaline, BasicPerceptron
from neural_networks.neuron.plotters import neuron_plotter

import time

ADALINE = 'Adaline'
BASIC_PERCEPTRON = 'BasicPerceptron'


def learn(neuron_class, params, is_to_print=False, is_to_plot=False):
    neuron = None
    if neuron_class == ADALINE:
        neuron = Adaline(**params)
    elif neuron_class == BASIC_PERCEPTRON:
        neuron = BasicPerceptron(**params)
    neuron.learn()
    if is_to_print:
        print(neuron)
    if is_to_plot:
        neuron_plotter.plot(*neuron.get_plot_data())
    return neuron


def run(neuron_class, activation_func, params, times, is_to_plot=False, is_to_print=False):
    params['activation_func'] = activation_func
    parameter_preparer.preapre(params=params)
    start = time.time()
    for i in range(times):
        learn(neuron_class, params, is_to_print=is_to_print, is_to_plot=is_to_plot)
    stop = time.time()
    return stop - start


def run_avg_epochs(neuron_class, activation_func, params, times):
    params['activation_func'] = activation_func
    parameter_preparer.preapre(params=params)
    accumulator = 0
    if neuron_class == ADALINE:
        times_passed = 0
        for i in range(times):
            neuron = learn(neuron_class, params)
            if neuron.result == 'adaline correct':
                accumulator += neuron.epochs
                # print(neuron)
                times_passed += 1
        print("pass rate: {}".format(str(times_passed / times)))
    elif neuron_class == BASIC_PERCEPTRON:
        for i in range(times):
            neuron = learn(neuron_class, params)
            accumulator += neuron.epochs
    return accumulator / times


# def run_avg_epochs(neuron_class, activation_func, params, times):
#     params['activation_func'] = activation_func
#     parameter_preparer.preapre(params=params)
#     accumulator = 0
#     for i in range(times):
#         neuron = learn(neuron_class, params)
#         accumulator += neuron.epochs
#     return accumulator / times


def run_many_params(neuron_class, activation_func, list_params, times=1000):
    metadata = list_params[0]
    epochs = [run_avg_epochs(neuron_class, activation_func, params, times) for params in list_params[1:]]
    return metadata, epochs
