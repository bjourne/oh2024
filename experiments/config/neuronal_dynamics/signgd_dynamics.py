from snn_signgd import setup
from snn_signgd.functional_config import FunctionalConfig, Munch
from snn_signgd.neuronal_dynamical_system.spikingjelly.generalization import construct_spiking_neurons_for_operators
from snn_signgd.neuronal_dynamical_system.spikingjelly.generalization.sgd import SGDModule

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
