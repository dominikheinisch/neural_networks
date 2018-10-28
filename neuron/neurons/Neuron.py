from abc import ABC, abstractmethod
import numpy as np


class Neuron(ABC):
    def __init__(self, **kwargs):
        self.learning_param = kwargs['learning_param']
        self.learning_data = kwargs['learning_pairs']
        self.weights = self.calc_weights(scope=kwargs['scope'])
        self.epochs = 0

    @staticmethod
    def calc_weights(scope):
        return np.random.uniform(low=scope[0], high=scope[1], size=3)

    def learn(self):
        input_pairs = list(self.learning_data.keys())
        np.random.shuffle(input_pairs)
        while not all([self.is_output_correct(pair) for pair in input_pairs]):
            np.random.shuffle(input_pairs)
            self.epochs += 1

    @abstractmethod
    def is_output_correct(self, pair):
        pass

    def __str__(self):
        return 'epochs: {0}\nlast weights: {1}\nlearning_param: {2}\ndata: {3}\n'\
            .format(self.epochs, self.weights, self.learning_param, self.learning_data)