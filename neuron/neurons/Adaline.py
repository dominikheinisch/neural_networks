from neural_networks.neuron.neurons import Neuron

import numpy as np

CORRECT = 'adaline correct'
INFINITY_LOOP = 'adaline fail, wrong parameters, cause infinity loop'
THRESHOLD_TOO_HUGE = 'adaline fail, threshold too huge'


class Adaline(Neuron):
    def __init__(self, **kwargs):
        super(Adaline, self).__init__(**kwargs)
        self.threshold = kwargs['threshold']
        self.err = []
        self.is_infinity_loop = False
        self.is_correct_result = None
        self.result = None

    def learn(self):
        self.learn_weights()
        self.check_is_correct_result()
        self.check_result()

    def learn_weights(self):
        input_pairs = list(self.learning_data.keys())
        np.random.shuffle(input_pairs)
        inf_loop_params = {'grow_counter': 0, 'repeat_counter': 0, 'errors_sum': [0.001, 0.001]}
        while not all([self.is_err_correct(pair, inf_loop_params) for pair in input_pairs]) \
                and not self.is_infinity_loop:
            np.random.shuffle(input_pairs)
            self.epochs += 1
            inf_loop_params['errors_sum'].append(0)

    def is_err_correct(self, input_pair, inf_loop_params):
        self.err.append(self.learning_data[input_pair][1] - self.weights @ self.learning_data[input_pair][0])
        self.weights += 2 * self.learning_param * self.err[-1] * self.learning_data[input_pair][0]
        inf_loop_params['errors_sum'][-1] += abs(self.err[-1])
        self.check_is_infinity_loop(inf_loop_params)
        return abs(self.err[-1]) < self.threshold

    def check_is_infinity_loop(self, inf_params):
        is_inf = np.inf == inf_params['errors_sum'][-1]
        if is_inf or self.is_growing(inf_params) or self.is_repeat_counter(inf_params) or \
                len(inf_params['errors_sum']) > 30:
            self.is_infinity_loop = True

    @staticmethod
    def is_growing(inf_params):
        inf_params['grow_counter'] = inf_params['grow_counter'] + 1 if abs(inf_params['errors_sum'][-1]) > \
                                                                       abs(inf_params['errors_sum'][-2]) else 0
        return inf_params['grow_counter'] > 4

    def is_repeat_counter(self, inf_params):
        last = inf_params['errors_sum'][-1]
        last_but_one = inf_params['errors_sum'][-2]
        is_repeat = abs((last - last_but_one) / last_but_one) < 0.01 * self.learning_param
        inf_params['repeat_counter'] = inf_params['repeat_counter'] + 1 if is_repeat else 0
        return inf_params['repeat_counter'] > 4

    def check_is_correct_result(self):
        input_pairs = list(self.learning_data.keys())
        inputs = np.array([self.learning_data[pair][0] for pair in input_pairs])
        outpus = np.array([self.learning_data[pair][1] for pair in input_pairs])
        prediction = np.array([self.activation_func(val) for val in inputs @ self.weights])
        self.is_correct_result = np.array_equal(prediction, outpus)

    def check_result(self):
        if self.is_infinity_loop:
            self.result = INFINITY_LOOP
        elif self.is_correct_result:
            self.result = CORRECT
        else:
            self.result = THRESHOLD_TOO_HUGE

    def get_plot_data(self):
        return self.learning_data.keys(), self.weights

    def __str__(self):
        return super().__str__() + \
               'threshold: {0}\nerrors: {1}\n4 last errors: {2}\nis_infinity_loop: {3}\n' \
               'is_correct_result: {4}\nresult: {5}\n'.format(
                   self.threshold, self.err, self.err[-4:], self.is_infinity_loop, self.is_correct_result, self.result)
