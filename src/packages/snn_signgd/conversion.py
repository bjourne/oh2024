import torch
from torch.utils.data import DataLoader

from .snn import SpikingNeuralNetwork
from .port_ann import porting

from snn_signgd.pretty_printer import print

def convert(
        ann_model,
        neuronal_dynamics,
        dynamics_type:str,
        n_time_steps:int,
        activation_scale_dataloader:DataLoader,
        max_activation_scale_iterations:int,
        scale_relu_with_max_activation:bool,
        sample_size = None
):
    assert dynamics_type in ['subgradient', 'signgd']

    print("<cyan> Starting ANN to SNN Conversion Process </cyan>")
    ann_model.eval()

    if sample_size is not None:
        sample = torch.randn(sample_size)
    else:
        sample, _ = next(iter(activation_scale_dataloader))

    sample = sample.to(next(ann_model.parameters()).device)

    print("<cyan> Porting ANN to Enable Conversion  </cyan>")
    snn_compatible_ann_model = porting(
        ann_model,
        activation_scale_dataloader,
        max_activation_scale_iterations,
        scale_relu_with_max_activation
    )

    print("<cyan> Converting ANN to SNN </cyan>")
    snn_model = SpikingNeuralNetwork(
        snn_compatible_ann_model,
        neuronal_dynamics,
        n_time_steps,
        dynamics_type,
        sample
    )

    print("<cyan> Finished ANN to SNN Conversion Process! </cyan>")

    return snn_model, snn_compatible_ann_model, sample
