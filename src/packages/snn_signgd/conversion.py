import torch
import torch.nn.functional as F

from snn_signgd.core.layer import multiply_inverse_of_square_root
from snn_signgd.graph_functional import pattern_matching_transform
from snn_signgd.graph_functional.transforms import replace_op, replace_ops_cases
from snn_signgd.port_ann import porting
#from spikingjelly.activation_based import functional
from spikingjelly.activation_based.functional import reset_net

from torch.nn import *
from tqdm import tqdm

torch.fx.wrap("multiply_inverse_of_square_root")

def _to_spiking_neuron_signgd(ann, config):
    model, log = pattern_matching_transform(
        ann,
        patterns = [
            (torch.relu,), (F.relu,), (ReLU,),
            (LeakyReLU,),
            (GELU,),
            (torch.maximum,),
            (multiply_inverse_of_square_root,),
            (torch.square,),
            (torch.exp,),
            (torch.matmul,),
            (torch.div,),
            (torch.abs,),
        ],
        graph_transform = replace_ops_cases(
            dest_modules = (
                lambda : config.relu(step_mode='s', v_reset= None),
                lambda : config.leakyrelu(step_mode='s', v_reset= None),
                lambda : config.gelu(step_mode='s', v_reset= None),
                lambda : config.maxpool(step_mode='s', v_reset= None),
                lambda : config.mul_inverse_sqrt(step_mode='s', v_reset= None),
                lambda : config.square(step_mode='s', v_reset= None),
                lambda : config.exp(step_mode='s', v_reset= None),
                lambda : config.matmul(step_mode='s', v_reset= None),
                lambda : config.div(step_mode='s', v_reset= None),
                lambda : config.abs(step_mode='s', v_reset= None),
            ),
            cases = (
                [(torch.relu,), (F.relu,), (ReLU,)],
                [(LeakyReLU,)],
                [(GELU,)],
                [(torch.maximum,)],
                [(multiply_inverse_of_square_root,)],
                [(torch.square,)],
                [(torch.exp,)],
                [(torch.matmul,)],
                [(torch.div,)],
                [(torch.abs,)],
            ),
            inherit_kwargs = (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        inplace = False,
        verbose = True,
    )
    return model, log

def _to_spiking_neuron_subgradient(ann, config):
    model, log = pattern_matching_transform(
        ann,
        patterns = [(torch.relu,), (F.relu,), (nn.ReLU,)],
        graph_transform =  replace_op(
            lambda : config.neuron(step_mode='s', v_reset= None)
        ),
        inplace = False
    )
    return model, log

nonlinearity_to_spiking_neuron = {
    'signgd' : _to_spiking_neuron_signgd,
    'subgradient' : _to_spiking_neuron_subgradient
}

class SpikingNeuralNetwork(Module):
    def __init__(
            self, ann, config:dict,
            default_simulation_length:int,
            dynamics_type, sample_data
    ):
        super().__init__()

        self.simulation_length = default_simulation_length

        self.model, self.log_transform = \
            nonlinearity_to_spiking_neuron[dynamics_type](
                ann, config
            )

        corrections = config.correction(
            net = self.model, sample_data = sample_data
        )

        self.codec = config.codec(statistics = corrections)

    def forward(self, x, timestamps = []):
        if timestamps:
            simulation_length = max(timestamps)
            history = []
        else:
            simulation_length = self.simulation_length

        reset_net(self.model)
        if hasattr(self.codec, 'reset'):
            self.codec.reset()

        x_enc = self.codec.encode(x)

        y = 0.0
        for timestep in tqdm(range(1, simulation_length + 1)):
            x_enc_t = next(x_enc)
            y_enc_t = self.model(x_enc_t)
            y = self.codec.decode(y, y_enc_t, timestep )

            if timestep in timestamps:
                history.append(y.clone().detach())

        if timestamps:
            return y, history
        else:
            return y

def convert(
        ann,
        neuronal_dynamics,
        dynamics_type,
        n_time_steps,
        loader,
        n_scaling_iters,
        relu_scaling,
        sample_size = None
):
    assert dynamics_type in ['subgradient', 'signgd']

    print("<cyan> Starting ANN to SNN Conversion Process </cyan>")
    ann.eval()

    if sample_size is not None:
        sample = torch.randn(sample_size)
    else:
        sample, _ = next(iter(loader))

    sample = sample.to(next(ann.parameters()).device)

    snn_compatible_ann = porting(ann, loader, n_scaling_iters, relu_scaling)

    snn = SpikingNeuralNetwork(
        snn_compatible_ann,
        neuronal_dynamics,
        n_time_steps,
        dynamics_type,
        sample
    )

    print("<cyan> Finished ANN to SNN Conversion Process! </cyan>")

    return snn, snn_compatible_ann, sample
