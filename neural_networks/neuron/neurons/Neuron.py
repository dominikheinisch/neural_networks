import numpy as np


class Neuron:
    def __init__(self, **kwargs):
        # self.learn_param = kwargs['alpha']
        self.alpha = kwargs['alpha']
        self.invalid_output = kwargs['invalid_output']
        self.learning_pairs = kwargs['learning_pairs']
        self.threshold = kwargs['threshold']
        self.weights = self.calc_weights(scope=kwargs['scope'])
        self.epochs = 0

    def calc_weights(self, scope):
        return np.random.uniform(low=scope[0], high=scope[1], size=3)

    def learn(self):
        input_pairs = list(self.learning_pairs.keys())
        np.random.shuffle(input_pairs)
        while not all([self.is_output_correct(pair) for pair in input_pairs]):
            np.random.shuffle(input_pairs)
            self.epochs += 1

    def __str__(self):
        return '{0}\n{1}\n{2}\n'.format(self.epochs, self.weights, self.learning_pairs)
