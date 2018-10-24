from neural_networks.neuron.helpers import parameter_preparer
from neural_networks.neuron.neurons import Adaline, BasicPerceptron
from neural_networks.neuron.plotters import plotter

import time


def learn(neuron_class, params, is_to_print=False, is_to_plot=False):
    neuron = None
    if neuron_class == 'Adaline':
        neuron = Adaline(**params)
    elif neuron_class == 'BasicPerceptron':
        neuron = BasicPerceptron(**params)
    neuron.learn()
    if is_to_print:
        print(neuron)
    if is_to_plot:
        plotter.plot(*neuron.get_plot_data())


def run(neuron_class, activation_func, params, is_to_plot=False, is_to_print=False, times=1000):
    params['activation_func'] = activation_func
    parameter_preparer.preapre(params=params)
    start = time.time()
    for i in range(times):
        learn(neuron_class, params, is_to_print=is_to_print, is_to_plot=is_to_plot)
    stop = time.time()
    return stop - start
