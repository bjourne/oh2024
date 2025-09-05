import torch
from torch.nn import *
from spikingjelly.activation_based import neuron
from .psychoactive_substance import Psychoactive

class BaseNeuron(neuron.BaseNode, Psychoactive):
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def neuronal_charge(self, x):
        raise NotImplementedError()

    def neuronal_fire(self):
        raise NotImplementedError()

    def neuronal_reset(self, spike):
        raise NotImplementedError()

class BaseCodec(Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def encodings(self) -> dict:
        raise NotImplementedError()

    def encode(self, x):
        encodings = self.encodings()
        assert self.choice in encodings.keys(), list(encodings.keys())
        return encodings[self.choice](x)

    def decode(self, state, spikes, timestep:int):
        raise NotImplementedError()
