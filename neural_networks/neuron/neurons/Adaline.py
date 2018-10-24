from neural_networks.neuron.neurons import Neuron

import numpy as np


class Adaline(Neuron):
    def __init__(self, **kwargs):
        super(Adaline, self).__init__(**kwargs)

    def learn(self):
        input_pairs = list(self.learning_pairs.keys())
        np.random.shuffle(input_pairs)
        while not all([self.is_output_correct(pair) for pair in input_pairs]):
            np.random.shuffle(input_pairs)
            self.epochs += 1

    def is_output_correct(self, input_pair):
        err = self.learning_pairs[input_pair][1] - self.activation_func(input_pair)
        if err == 0:
            return True
        else:
            self.weights += self.alpha * err * self.learning_pairs[input_pair][0]
            return False

    def activation_func(self, input_pair):
        return 1 if self.weights @ self.learning_pairs[input_pair][0] > 0 else self.invalid_output

    def get_plot_data(self):
        return self.learning_pairs.keys(), self.weights

    # def fun(self, learning_pair):
    #     net = self.weights @ self.learning_pairs[learning_pair][0]
    #     err = self.learning_pairs[learning_pair][1] - net
    #     self.weights += self.alpha * err * self.learning_pairs[learning_pair][0]
    #     # print(net, err)
    #     print(self.weights)
    #     return False