import torch
from spikingjelly.activation_based import neuron
from ...psychoactive_substance import Psychoactive
from itertools import count
from munch import Munch
from torch import nn
from torch.nn import *

def relocate(src, dst):
    if torch.is_tensor(src):
        src = src.to(dst.device)
    return src

def sign(spike):
    return 2 * spike - 1

def make_opt():
    return SGDModule(lr = 0.15, inplace = False)

def make_sched():
    return ExponentialScheduler() # gamma = 0.95, eager_evaluation = True)

# class BaseNeuron(neuron.BaseNode, Psychoactive):
#     def __init__(self, **kwargs):
#         raise NotImplementedError()

#     def reset(self):
#         raise NotImplementedError()

#     def neuronal_charge(self, x):
#         raise NotImplementedError()

#     def neuronal_fire(self):
#         raise NotImplementedError()

#     def neuronal_reset(self, spike):
#         raise NotImplementedError()

class ExponentialScheduler:
    def __init__(self): #, gamma: float, eager_evaluation: bool = False):
        self.gamma = 0.95
        self.lr = None
        #self.eager_evaluation = True

    def reset(self, moduleoptimizer):
        self.moduleoptimizer = moduleoptimizer
        if self.lr is None:
            self.lr = self.moduleoptimizer.config['lr']
        else:
            self.moduleoptimizer.config['lr'] = self.lr

        #if self.eager_evaluation:
        M = 1024
        self.lrs = [self.lr * (self.gamma ** i) for i in range(M)]
        self.timestep = 1

    def schedule(self):
        #if self.eager_evaluation:
        self.moduleoptimizer.config['lr'] = self.lrs[self.timestep]
        # else:
        #     self.moduleoptimizer.config['lr'] *= self.gamma
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

        assert momentum == 0

        # if momentum != 0:
        #     if buf is None:
        #         buf = torch.clone(d_p).detach()
        #     else:
        #         buf = momentum * buf + (1-dampening) * d_p

        #     if nesterov:
        #         d_p = d_p + buf * momentum
        #     else:
        #         d_p = buf

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

class Neuron(neuron.BaseNode, Psychoactive):
    def __init__(self, spike_mechanism, **kwargs):
        neuron.BaseNode.__init__(self, **kwargs)
        Psychoactive.__init__(self)

        self.optimizer_input = make_opt()
        self.optimizer_output = make_opt()
        self.lr_scheduler_input = make_sched()
        self.lr_scheduler_output = make_sched()

        self.spike_mechanism = spike_mechanism

        self.bias = 0.0
        self.reset()

    def reset(self):

        self.optimizer_input.reset(initializer = self.bias)
        self.optimizer_output.reset(initializer = 0.0)
        self.lr_scheduler_input.reset(moduleoptimizer = self.optimizer_input)
        self.lr_scheduler_output.reset(moduleoptimizer = self.optimizer_output)

        self.v = 0.0

    def neuronal_charge(self, x):
        gradient = 2 * x - relocate(self.offset, x)

        self.x = self.optimizer_input.step(grad = gradient)
        self.lr_scheduler_input.schedule()

    def neuronal_fire(self):
        return self.spike_mechanism(self)

    def neuronal_reset(self, spike):
        gradient = 2 * spike - 1

        self.v = self.optimizer_output.step(gradient)
        self.lr_scheduler_output.schedule()

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

class Codec(BaseCodec, Module):
    def __init__(self, choice:str, statistics:dict = None):
        Module.__init__(self)
        self.choice = choice

        self.optimizer_enc = make_opt()
        self.optimizer_dec = make_opt()
        self.lr_scheduler_enc = make_sched()
        self.lr_scheduler_dec = make_sched()

        if statistics is None:
            self.offset, self.bias = 1.0, 0.0
        else:
            for key, value in statistics.items():
                self.register_buffer(key, value)

        self.reset()

    def encodings(self) -> dict:
        return {
            'float': self.encode_template(self.grad_float),#self.float_encode,
            'spike': self.encode_template(self.grad_spike),#self.spike_encode,
            #'stochastic_spike': self.encode_template(self.grad_stochastic_spike),
        }

    def reset(self):
        self.optimizer_enc.reset(initializer = 0.0)
        self.optimizer_dec.reset(initializer = self.bias)
        self.lr_scheduler_enc.reset(moduleoptimizer = self.optimizer_enc)
        self.lr_scheduler_dec.reset(moduleoptimizer = self.optimizer_dec)
        #print("Codec reset called")

    def grad_float(self, y, x):
        gradient = y - x # Loss function L(y;x) = \Vert y - x \Vert_2^2
        spike = 0.5 * (1 + gradient)
        return gradient, spike

    def grad_spike(self, y, x):
        spike = torch.heaviside( y - x , torch.zeros_like(x)) # Loss function L(y;x) = \Vert y - x \Vert_2^2
        gradient = sign(spike)
        return gradient, spike

    # def grad_stochastic_spike(self, y, x):
    #     spike = torch.bernoulli(torch.sigmoid( (y - x) )) # Loss function L(y;x) = \Vert y - x \Vert_2^2
    #     gradient = sign(spike)
    #     return gradient, spike

    def encode_template(self, gradient_fn):
        def encode_fn(x):
            f = 0.0
            for timestep in count(1):
                gradient, spike = gradient_fn(f,x)

                f = self.optimizer_enc.step(grad = gradient)
                self.lr_scheduler_enc.schedule()

                yield spike
        return encode_fn

    def decode(self, state:torch.Tensor, spikes:torch.Tensor, timestep:int) -> torch.Tensor:
        gradient = 2 * spikes - relocate(self.offset, spikes) # W * x(t)

        y = self.optimizer_dec.step(grad = gradient)
        self.lr_scheduler_dec.schedule()

        return y
