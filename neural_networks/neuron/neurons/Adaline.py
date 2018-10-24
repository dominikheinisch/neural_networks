from neural_networks.neuron.neurons import Neuron

class Adaline(Neuron):
    def __init__(self, **kwargs):
        super(Adaline, self).__init__(**kwargs)