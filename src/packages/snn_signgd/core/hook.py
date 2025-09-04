
import torch
from torch import nn
import torch.nn.functional as F

import os
from typing import Callable, List, Any
from collections import defaultdict
from munch import Munch

import numpy as np

from spikingjelly.activation_based import neuron

import snn_signgd.dictfs as dictfs

from ..neuronal_dynamical_system.spikingjelly.template import BaseNeuron, BaseCodec
from ..neuronal_dynamical_system.spikingjelly.psychoactive_substance import stimulant, depressant, Psychoactive
from ..neuronal_dynamical_system.spikingjelly.dtype import TensorPair


class hook_context(object):
    def __init__(self, hook_fn: Callable, module:nn.Module = None) -> None:
        self.module = module
        self.hook_fn = hook_fn

    def __enter__(self) -> None:
        if self.module is None:
            hook = nn.modules.module.register_module_forward_hook(
                self.hook_fn
            )
        else:
            hook = self.module.register_forward_hook(self.hook_fn)
        self.hooks = [hook]
        return self

    def __exit__(self, exc_type:Any, exc_value:Any, traceback:Any) -> None:
        for hook in self.hooks:
            hook.remove()

def running_average(state, input, timestep):
    return timestep / (1.0 + timestep) * state + 1.0 / (1.0 + timestep) * input

def activation_stats_hook(module, input, output):

    if not hasattr(module, 'activation_stats'):
        module.activation_stats = defaultdict(lambda: None)
        module.activation_stats['timestep'] = 0

    #print("Collect Activation Stats:",module.activation_stats)

    time_dependent_stats = ['mean', 'square_moment']
    for name, stat_func, update_func in [
            ('max',lambda x: torch.max(x, dim = 0)[0], torch.maximum),
            ('min',lambda x: torch.min(x, dim = 0)[0], torch.minimum),
            ('mean',lambda x: torch.mean(x, dim = 0), running_average),
            ('square_moment',lambda x: torch.mean(torch.square(x), dim = 0), running_average),
    ]:
        for indicator, x in zip(['input', 'output'], [input[0],output]):
            if not isinstance(x, torch.Tensor): continue

            x_np = x.detach()

            path = os.path.join(indicator,name)
            value = stat_func(x_np)

            if module.activation_stats[path] is None:
                module.activation_stats[path] = value
            else:
                if name in time_dependent_stats:
                    module.activation_stats[path] = update_func(
                        module.activation_stats[path], value,
                        timestep = module.activation_stats['timestep']
                    )
                else:
                    module.activation_stats[path] = update_func(
                        module.activation_stats[path], value
                    )
    module.activation_stats['timestep'] += 1
    return
