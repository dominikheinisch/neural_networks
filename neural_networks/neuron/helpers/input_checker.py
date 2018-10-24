from sys import argv

def check_input():
    if not (argv[2] == 'Adaline' or argv[2] == 'BasicPerceptron'):
        raise ValueError('wrong neuron_type')
    if not (argv[3] == 'binary' or argv[3] == 'bipolar'):
        raise ValueError('wrong activation_func')