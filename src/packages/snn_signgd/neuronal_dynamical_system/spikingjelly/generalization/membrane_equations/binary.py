import torch

from spikingjelly.activation_based import neuron
from ...psychoactive_substance import Psychoactive
from ...template import BaseNeuron, BaseCodec
from .unary import Neuron
from ...dtype import TensorPair
from itertools import count
from munch import Munch
from torch.nn import Module

def relocate(src, dst):
    if torch.is_tensor(src):
        src = src.to(dst.device)
    return src

def sign(spike):
    return 2 * spike - 1

class NeuronWrapper(Module):
    def __init__(self, **neuronal_dynamics_kwargs):
        super().__init__()
        #print("Kwargs:", neuronal_dynamics_kwargs)
        self.neuron = Neuron(**neuronal_dynamics_kwargs)
    def forward(self, x,y):
        #raise NotImplementedError()
        pair = TensorPair(x, y)
        output = self.neuron(pair).to(x)
        return output
