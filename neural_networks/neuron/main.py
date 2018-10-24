from neural_networks.neuron.simulations import simulation
from neural_networks.neuron.helpers import input_checker

from sys import argv


if __name__ == "__main__":
    learning_param = 0.05
    scope = (-1.0, 1.0)
    params = {'learning_param': learning_param, 'scope': scope, 'threshold': 1.1}

    # bias = 1
    # params['bias'] = bias
    # learning_pairs = {(0, 0): 0, (1, 0): 0, (1, 1): 1, (0, 1): 0}
    # params['learning_pairs'] = learning_pairs

    if len(argv) == 1:
        # time = simulation.run('BasicPerceptron', 'binary', params, is_to_plot=True)
        # simulation.run('BasicPerceptron', 'bipolar', params, is_to_plot=True)
        # simulation.run('Adaline', 'binary', params, is_to_plot=True)
        simulation.run('Adaline', 'bipolar', params, is_to_plot=True)
    else:
        input_checker.check_input()
        time = simulation.run(argv[2], argv[3], params, is_to_plot=True)
    print(time)



