from neural_networks.neuron.neurons import Neuron

import numpy as np


class BasicPerceptron(Neuron):
    def __init__(self, **kwargs):
        super(BasicPerceptron, self).__init__(**kwargs)

    def learn_weights(self):
        input_pairs = list(self.learning_data.keys())
        np.random.shuffle(input_pairs)
        while not all([self.is_err_correct(pair) for pair in input_pairs]):
            np.random.shuffle(input_pairs)
            self.epochs += 1

    def is_err_correct(self, input_pair):
        err = self.learning_data[input_pair][1] - self.activation_func(self.learning_data[input_pair][0] @ self.weights)
        if err == 0:
            return True
        else:
            self.weights += self.learning_param * err * self.learning_data[input_pair][0]
            return False

    def get_plot_data(self):
        return self.learning_data.keys(), self.weights
