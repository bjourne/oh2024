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

def spike_mechanism_gelu(neuron):
    y = neuron.v

    y = y.to(neuron.x)
    spike = torch.heaviside(
        y - neuron.x * torch.sigmoid(1.702 * neuron.x),
        torch.zeros_like(y)
    )
    return spike


def spike_mechanism_leakyrelu(n, neg_slope):
    y = n.v

    condition = n.x >= 0
    trueval = torch.heaviside( y - n.x , torch.tensor([0.0]))
    falseval = torch.heaviside( y - neg_slope * n.x, torch.tensor([0.0]))
    spike = condition * trueval + torch.logical_not(condition) * falseval

    return spike


def spike_mechanism_multiply(neuron):
    y = neuron.v

    x1, x2  = neuron.x.x, neuron.x.y
    y = y.to(x1)
    spike = (y >= torch.mul(x1,x2)).to(y)

    return spike

def spike_mechanism_relu(neuron):
    y = neuron.v

    condition = neuron.x >= 0
    trueval = y >= neuron.x
    falseval = y >= 0

    # (neuron.x >= 0 && neuron.v >= neuron.x) || (!(neuron.x >= 0) && neuron.v >= 0)
    # Like Corollary 5.4 I think
    # Where is .v  and .x updated?
    spike = torch.logical_or(
        torch.logical_and(condition, trueval),
        torch.logical_and(torch.logical_not(condition), falseval)
    ).to(neuron.x)

    return spike

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


def construct_spiking_neurons_for_operators(
    moduleoptimizer_cfg, lr_scheduler_cfg
):
    return Munch(
        relu = FunctionalConfig(
                module = Neuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_relu,
            ),
        leakyrelu = FunctionalConfig(
                module = Neuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = partial(spike_mechanism_leakyrelu, neg_slope = 0.1),
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
                module = Neuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_gelu,
            ),
        square = FunctionalConfig(
                module = Neuron,
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
                module = Neuron,
                submodules = Munch(
                    optimizer_input = moduleoptimizer_cfg,
                    optimizer_output = moduleoptimizer_cfg,
                    lr_scheduler_input = lr_scheduler_cfg,
                    lr_scheduler_output = lr_scheduler_cfg,
                ),
                spike_mechanism = spike_mechanism_exp,
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


class ExponentialScheduler:
    def __init__(self, gamma: float, eager_evaluation: bool = False):
        self.gamma = gamma
        self.lr = None
        self.eager_evaluation = eager_evaluation

    def reset(self, moduleoptimizer):
        self.moduleoptimizer = moduleoptimizer
        if self.lr is None:
            self.lr = self.moduleoptimizer.config['lr']
        else:
            self.moduleoptimizer.config['lr'] = self.lr

        if self.eager_evaluation:
            M = 1024
            self.lrs = [self.lr * (self.gamma ** i) for i in range(M)]
        self.timestep = 1

    def schedule(self):
        if self.eager_evaluation:
            self.moduleoptimizer.config['lr'] = self.lrs[self.timestep]
        else:
            self.moduleoptimizer.config['lr'] *= self.gamma
        self.timestep += 1

class SGDModule:
    def __init__(
            self, lr=1e-3, momentum=0, dampening=0,
            weight_decay=0, nesterov=False, inplace = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.config = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
            inplace = inplace
        )

    def reset(self, initializer):
        self.state = {
            'param': initializer,
            'momentum_buffer': None
        }

    def step(self, grad):
        state = self.sgd(
            param = self.state['param'],
            grad = grad,
            momentum_buffer = self.state['momentum_buffer'],
            weight_decay=self.config['weight_decay'],
            momentum=self.config['momentum'],
            lr=self.config['lr'],
            dampening=self.config['dampening'],
            nesterov=self.config['nesterov'],
            inplace = self.config['inplace'])

        self.state = state
        return state['param']

    def sgd(self,
            param, grad, momentum_buffer,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool,
            inplace: bool
    ):
        d_p = grad
        buf = momentum_buffer


        if weight_decay != 0:
            if inplace:
                d_p.add_(param, alpha=weight_decay) # Inplace operation
            else:
                d_p = d_p + param * weight_decay # Not an inplace operation

        if momentum != 0:
            if buf is None:
                buf = torch.clone(d_p).detach()
            else:
                buf = momentum * buf + (1-dampening) * d_p

            if nesterov:
                d_p = d_p + buf * momentum
            else:
                d_p = buf

        if inplace:
            d_p.multiply_(-lr)
            param = d_p.add(param)
        else:
            #param = torch.add(param, d_p, alpha = -lr) #
            param = param - lr * d_p

        state = {
            'param': param,
            'momentum_buffer': buf
        }
        return state

neuronal_dynamics_per_ops = construct_spiking_neurons_for_operators(
    moduleoptimizer_cfg = FunctionalConfig(
        module = SGDModule,
        lr = 0.15,
        inplace = False,
    ),
    lr_scheduler_cfg = FunctionalConfig(
        module = ExponentialScheduler,
        gamma = 0.95,
        eager_evaluation = True,
    ),
)

config = Munch(
    dynamics_type = 'signgd',

    default_simulation_length = 32,
    max_activation_scale_iterations = 10,
    scale_relu_with_max_activation = True,

    neuronal_dynamics = neuronal_dynamics_per_ops,

    setup = setup,
)
