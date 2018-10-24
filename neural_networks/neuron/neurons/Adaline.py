from neural_networks.neuron.neurons import Neuron

import numpy as np


class Adaline(Neuron):
    def __init__(self, **kwargs):
        super(Adaline, self).__init__(**kwargs)
        self.threshold = kwargs['threshold']

    def learn(self):
        input_pairs = list(self.learning_data.keys())
        np.random.shuffle(input_pairs)
        while not all([self.is_output_correct(pair) for pair in input_pairs]):
            np.random.shuffle(input_pairs)
            self.epochs += 1

    def is_output_correct(self, input_pair):
        err = self.learning_data[input_pair][1] - self.weights @ self.learning_data[input_pair][0]
        print(err)
        self.weights += 2 * self.learning_param * err * self.learning_data[input_pair][0]
        # print(self.weights)
        return abs(err) < self.threshold

    def get_plot_data(self):
        return self.learning_data.keys(), self.weights
