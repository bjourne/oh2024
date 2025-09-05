from functools import partial
from snn_signgd import setup
from snn_signgd.functional_config import FunctionalConfig, Munch
from snn_signgd.neuronal_dynamical_system.spikingjelly.generalization.membrane_equations import Codec, correction, BinaryNeuron
from snn_signgd.neuronal_dynamical_system.spikingjelly.generalization.membrane_equations.unary import Neuron

import torch

def spike_mechanism_square(neuron):
    y = neuron.v
    spike = torch.heaviside(y - neuron.x ** 2, neuron.x) # TODO
    return spike

def spike_mechanism_exp(neuron):
    y = neuron.v
    spike = (y >= torch.exp(neuron.x)).to(y)
    return spike


def spike_mechanism_div(neuron):
    y = neuron.v
    x1, x2  = neuron.x.x, neuron.x.y

    y = y.to(x1)
    spike = torch.heaviside(y - torch.divide(x1,x2), torch.zeros_like(x1)) # TODO

    return spike


def spike_mechanism_maximum(neuron):
    y = neuron.v
    x1, x2  = neuron.x.x, neuron.x.y

    y = y.to(x1)
    spike = torch.heaviside(y - torch.max(x1, x2), torch.zeros_like(x1))

    return spike

def spike_mechanism_gelu(n):
    y = n.v

    y = y.to(n.x)
    spike = torch.heaviside(
        y - n.x * torch.sigmoid(1.702 * n.x),
        torch.zeros_like(y)
    )
    return spike


def spike_mechanism_leakyrelu(n, neg_slope):
    y = n.v

    condition = n.x >= 0
    trueval = torch.heaviside(n.v - n.x , torch.tensor([0.0]))
    falseval = torch.heaviside(n.v - neg_slope * n.x, torch.tensor([0.0]))
    spike = condition * trueval + torch.logical_not(condition) * falseval

    return spike


def spike_mechanism_multiply(n):
    #y = n.v

    # x1, x2  = n.x.x, n.x.y
    # y = y.to(x1)
    # spike = (n.v >= torch.mul(n.x.x, n.x.y)).to(y)

    return n.v >= torch.mul(n.x.x, n.x.y)

def spike_mechanism_relu(n):
    #y = n.v

    #condition = n.x >= 0
    #trueval = y >= n.x
    #falseval = y >= 0

    # (n.x >= 0 && n.v >= n.x) || (!(n.x >= 0) && n.v >= 0)
    # Like Corollary 5.4 I think
    # Where is .v  and .x updated?
    return torch.logical_or(
        torch.logical_and(n.x >= 0, n.v >= n.x),
        torch.logical_and(torch.logical_not(n.x >= 0), n.v >= 0)
    ).to(n.x)
    #return spike

def spike_mechanism_square(neuron):
    y = neuron.v
    spike = (y >= neuron.x ** 2).to(y)
    return spike


def spike_multiply_inverse_of_square_root(neuron):
    y = neuron.v
    x1, x2  = neuron.x.x, neuron.x.y

    y = y.to(x1)
    spike = torch.heaviside( torch.sqrt(x2) * y - x1 , torch.zeros_like(x1))

    return spike


def construct_spiking_neurons_for_operators():
    return Munch(
        relu = FunctionalConfig(
                module = Neuron,
                spike_mechanism = spike_mechanism_relu,
            ),
        leakyrelu = FunctionalConfig(
                module = Neuron,
                spike_mechanism = partial(spike_mechanism_leakyrelu, neg_slope = 0.1),
            ),
        maxpool = FunctionalConfig(
                module = BinaryNeuron,
                spike_mechanism = spike_mechanism_maximum,
            ),
        gelu = FunctionalConfig(
                module = Neuron,
                spike_mechanism = spike_mechanism_gelu,
            ),
        square = FunctionalConfig(
                module = Neuron,
                spike_mechanism = spike_mechanism_square,
            ),
        mul_inverse_sqrt = FunctionalConfig(
                module = BinaryNeuron,
                spike_mechanism = spike_multiply_inverse_of_square_root,
            ),
        exp = FunctionalConfig(
                module = Neuron,
                spike_mechanism = spike_mechanism_exp,
            ),
        div = FunctionalConfig(
                module = BinaryNeuron,
                spike_mechanism = spike_mechanism_div,
            ),
        codec = FunctionalConfig(
                module = Codec,
                choice = 'float'
            ),
        correction = FunctionalConfig(
            module = correction,
        ),
    )

neuronal_dynamics_per_ops = construct_spiking_neurons_for_operators()

config = Munch(
    dynamics_type = 'signgd',

    default_simulation_length = 32,
    max_activation_scale_iterations = 10,
    scale_relu_with_max_activation = True,

    neuronal_dynamics = neuronal_dynamics_per_ops,

    setup = setup,
)
