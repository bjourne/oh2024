import torch
import torch.nn.functional as F

def spike_mechanism_abs(neuron):
    y = neuron.v

    y = y.to(neuron.x)
    spike = torch.heaviside(
        y - neuron.x * torch.sign(neuron.x),
        torch.zeros_like(y)
    )
    return spike
