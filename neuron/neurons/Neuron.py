from abc import ABC, abstractmethod
import numpy as np


class Neuron(ABC):
    def __init__(self, **kwargs):
        self.invalid_output = kwargs['invalid_output']
        self.learning_param = kwargs['learning_param']
        self.learning_data = kwargs['learning_pairs']
        self.weights = self.calc_weights(scope=kwargs['scope'])
        self.begin_weights = self.weights
        self.epochs = 0

    @staticmethod
    def calc_weights(scope):
        return np.random.uniform(low=scope[0], high=scope[1], size=3)

    def learn(self):
        self.learn_weights()

    @abstractmethod
    def learn_weights(self, pair):
        pass

    @abstractmethod
    def is_err_correct(self, pair):
        pass

    def activation_func(self, value):
        return 1 if value > 0 else self.invalid_output

    def __str__(self):
        return 'epochs: {0}\nbegin weights: {1}\nlast weights: {2}\nlearning_param: {3}\ndata: {4}\n'\
            .format(self.epochs,self.begin_weights, self.weights, self.learning_param, self.learning_data)
