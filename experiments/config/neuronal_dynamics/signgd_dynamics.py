from snn_signgd import setup
from snn_signgd.functional_config import FunctionalConfig, Munch
from snn_signgd.neuronal_dynamical_system.spikingjelly.generalization import construct_spiking_neurons_for_operators

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
