from neural_networks.neuron.simulations import simulation

import numpy as np
import time


def transform_learning_pairs(params, bias):
    return {x: (np.array([bias, *x]), params[x]) for x in params}


if __name__ == "__main__":
    alpha = 0.1
    bias = 1
    learning_pairs = {(0, 0): 0, (1, 0): 0, (1, 1): 1, (0, 1): 0}
    scope = (-0.1, 0.1)
    # scope = (0, 0)
    params = {'temp': 't', 'alpha': alpha, 'bias': bias, 'scope': scope, 'threshold': 0.4,
              'learning_pairs': transform_learning_pairs(learning_pairs, bias)}

    start = time.time()
    for i in range(10000):
        # simulation.simulate(params)
        simulation.simulate(params, is_to_plot=True)
    stop = time.time()
    print(stop-start)
