from neural_networks.neuron.io import read

import matplotlib.pyplot as plt
import re


def get_params_form_filename(filename):
    return re.search('_result_(.*).json', filename).group(1)


def scope_x(metadata, x_vec):
    if metadata == 'scope':
        return [(abs(x[0]) + abs(x[1])) / 2 for x in x_vec]
    else:
        return x_vec


def set_plt_data(plt, files, x_name):
    plt.ylabel('avg epochs')
    plt.xlabel(x_name)
    names = [get_params_form_filename(f) for f in files]
    plt.legend(names, loc='upper right')
    plt.show()


def plot_scope(files):
    x_name = None
    x_vec = None
    for f in files:
        (metadata, x_vec), y_vec = read(f)
        x_name = metadata
        scope_x_ = scope_x(metadata, x_vec)
        plt.plot(scope_x_, y_vec)
    plt.xticks(scope_x_, x_vec)
    set_plt_data(plt, files, x_name)


def plot_default(files):
    x_name = None
    for f in files:
        (metadata, x_vec), y_vec = read(f)
        x_name = metadata
        plt.plot(x_vec, y_vec)
    set_plt_data(plt, files, x_name)


def plot(files):
    if any('scope' in f for f in files):
        plot_scope(files)
    else:
        plot_default(files)


if __name__ == "__main__":
    filename1 = '../data/scope_(-1, 1)_(-0.2, 0.2)_result_BasicPerceptron_binary.json'
    filename2 = '../data/scope_(-1, 1)_(-0.2, 0.2)_result_BasicPerceptron_bipolar.json'
    plot([filename1, filename2])
    filename1 = '../data/learning_param_0.01_0.5_result_BasicPerceptron_binary.json'
    filename2 = '../data/learning_param_0.01_0.5_result_BasicPerceptron_bipolar.json'
    plot([filename1, filename2])
