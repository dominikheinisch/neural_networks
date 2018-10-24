from neural_networks.neuron.helpers import parameter_preparer
from neural_networks.neuron.simulations import simulation

import time


if __name__ == "__main__":
    alpha = 0.1
    # activation_func = 'binary'
    activation_func = 'bipolar'
    bias = 1
    learning_pairs = {(0, 0): 0, (1, 0): 0, (1, 1): 1, (0, 1): 0}
    scope = (-0.1, 0.1)
    # scope = (0, 0)
    params = {'activation_func': activation_func, 'alpha': alpha, 'bias': bias, 'scope': scope, 'threshold': 0.4,
              'learning_pairs': learning_pairs}
    params = parameter_preparer.preapre(params=params)
    start = time.time()
    for i in range(10000):
        # simulation.simulate(params)
        simulation.simulate(params, is_to_plot=True)
    stop = time.time()
    print(stop-start)
