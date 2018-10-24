from neural_networks.neuron.neurons import Adaline, BasicPerceptron
from neural_networks.neuron.plotters import plotter


def simulate(params, is_to_plot=False):
    neuron = BasicPerceptron(**params)
    neuron.learn()
    if is_to_plot:
        plotter.plot(*neuron.get_plot_data())

