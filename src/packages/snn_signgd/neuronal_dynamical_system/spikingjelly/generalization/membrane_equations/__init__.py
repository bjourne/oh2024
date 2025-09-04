from .binary import NeuronWrapper as BinaryNeuron
from .unary import Neuron as UnaryNeuron
from .unary import Codec

import torch
from torch import nn
from ...psychoactive_substance import stimulant, depressant
from munch import Munch
from tqdm import tqdm
from snn_signgd.system_optimizations import reduce_duplicates

def correction(net, sample_data):
    with stimulant(net) as S:
        excited_output = net(S.excite(sample_data[0:1]))
    with depressant(net) as D:
        depressed_output = net(D.depress(sample_data[0:1]))

    for m in tqdm(net.modules(), desc = "Dynamics Correction", total = len(list(net.modules()))):
        if hasattr(m, 'activations'):
            offset = m.activations["excited"] - m.activations["depressed"] \
                + 2 * m.activations["depressed"]
            bias = m.activations["depressed"]

            offset = reduce_duplicates(offset[0])
            bias = reduce_duplicates(bias[0])

            del(m.activations)
            del(m.v)
            m.offset = offset
            m.bias = bias

    statistics = Munch(excited = excited_output.clone(), depressed = depressed_output.clone())
    offset = statistics["excited"] - statistics["depressed"] + 2 * statistics["depressed"]
    bias = statistics["depressed"]

    offset = reduce_duplicates(offset[0])
    bias = reduce_duplicates(bias[0])

    del(statistics)

    return Munch(offset = offset, bias = bias, input_init = None)
