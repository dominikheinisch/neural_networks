from neural_networks.neuron.simulations import simulation

import sys


def check_input():
    if not (sys.argv[2] == 'Adaline' or sys.argv[2] == 'BasicPerceptron'):
        raise ValueError('wrong neuron_type')
    if not (sys.argv[3] == 'binary' or sys.argv[3] == 'bipolar'):
        raise ValueError('wrong activation_func')


if __name__ == "__main__":
    alpha = 0.1

    scope = (-0.1, 0.1)
    # scope = (0, 0)
    params = {'alpha': alpha, 'scope': scope, 'threshold': 0.4}

    # bias = 1
    # params['bias'] = bias
    # learning_pairs = {(0, 0): 0, (1, 0): 0, (1, 1): 1, (0, 1): 0}
    # params['learning_pairs'] = learning_pairs

    if len(sys.argv) == 1:
        # time = simulation.run('BasicPerceptron', 'binary', params, is_to_plot=True)
        simulation.run('BasicPerceptron', 'bipolar', params, is_to_plot=True)
        # simulation.run('Adaline', 'bipolar', params, is_to_plot=True)
    else:
        check_input()
        time = simulation.run(sys.argv[2], sys.argv[3], params, is_to_plot=True)
    print(time)



