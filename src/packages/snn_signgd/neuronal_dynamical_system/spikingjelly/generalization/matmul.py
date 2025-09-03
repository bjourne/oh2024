import torch
from ..template import BaseNeuronWrapper
from .membrane_equations.binary import Neuron
from .membrane_equations.binary import TensorPair
from ..dtype import TensorPair

class MatMulNeuron(BaseNeuronWrapper):
    def __init__(self, **neuronal_dynamics_kwargs):
        super().__init__()
        self.neuron = Neuron(**neuronal_dynamics_kwargs)
    def forward(self, x, y):
        # Naive implementation
        num_right_tokens = y.shape[-1]

        repeat_count = num_right_tokens

        x = torch.unsqueeze(x, dim = -1)
        size = list(x.shape)
        size[-1] = repeat_count

        x, y = x.expand(*size), torch.unsqueeze(y, dim = -3)

        pair = TensorPair(x, y)

        output = self.neuron(pair).to(x)
        output = torch.sum(output, dim = -2)
        return output


def spike_mechanism_multiply(neuron):
    y = neuron.v

    x1, x2  = neuron.x.x, neuron.x.y
    y = y.to(x1)
    spike = (y >= torch.mul(x1,x2)).to(y)

    return spike

def spike_mechanism_square(neuron):
    y = neuron.v

    spike = torch.heaviside(y - neuron.x ** 2, neuron.x) # TODO

    return spike
