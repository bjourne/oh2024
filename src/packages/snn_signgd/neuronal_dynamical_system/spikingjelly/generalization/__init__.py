from functools import partial

from snn_signgd.functional_config import FunctionalConfig, Munch
from .membrane_equations import UnaryNeuron, Codec, correction, BinaryNeuron
from .leakyrelu import spike_mechanism_leakyrelu
from .maxpool import spike_mechanism_maximum
from .gelu import spike_mechanism_gelu
from .layernorm import spike_mechanism_square, spike_multiply_inverse_of_square_root
from .matmul import spike_mechanism_multiply, MatMulNeuron

import torch

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

def spike_mechanism_abs(neuron):
    y = neuron.v

    y = y.to(neuron.x)
    spike = torch.heaviside(
        y - neuron.x * torch.sign(neuron.x),
        torch.zeros_like(y)
    )
    return spike

def spike_mechanism_relu(neuron):
    y = neuron.v

    condition = neuron.x >= 0
    trueval = y >= neuron.x
    falseval = y >= 0

    # (neuron.x >= 0 && y >= neuron.x) || (!(neuron.x >= 0) && y >= 0)
    spike = torch.logical_or(
        torch.logical_and(condition, trueval),
        torch.logical_and(torch.logical_not(condition), falseval)
    ).to(neuron.x)

    return spike

def construct_spiking_neurons_for_operators(
    moduleoptimizer_cfg, lr_scheduler_cfg
):
    return Munch(
        relu = FunctionalConfig(
                module = UnaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_relu,
            ),
        leakyrelu = FunctionalConfig(
                module = UnaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = partial(spike_mechanism_leakyrelu, negative_slope = 0.1),
            ),
        maxpool = FunctionalConfig(
                module = BinaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_maximum,
            ),
        gelu = FunctionalConfig(
                module = UnaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_gelu,
            ),
        square = FunctionalConfig(
                module = UnaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_square,
            ),
        mul_inverse_sqrt = FunctionalConfig(
                module = BinaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_multiply_inverse_of_square_root,
            ),
        exp = FunctionalConfig(
                module = UnaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_exp,
            ),
        abs = FunctionalConfig(
                module = UnaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_abs,
            ),
        matmul = FunctionalConfig(
                module = MatMulNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_multiply,
            ),
        div = FunctionalConfig(
                module = BinaryNeuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_div,
            ),
        codec = FunctionalConfig(
                module = Codec,
                choice = 'float',
                submodules = Munch(
                    optimizer_enc = moduleoptimizer_cfg,
                    optimizer_dec = moduleoptimizer_cfg,
                    lr_scheduler_enc = lr_scheduler_cfg,
                    lr_scheduler_dec = lr_scheduler_cfg,
                ),
            ),
        correction = FunctionalConfig(
            module = correction,
        ),
    )
