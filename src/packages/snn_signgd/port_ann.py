import torch
import torch.nn.functional as F

from itertools import islice
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .graph_functional import pattern_matching_transform
from .graph_functional.transforms import replace_op, fuse_conv_bn, replace_ops_cases

from .core.layer import (
    ScaledOp,
    BinaryTreeMaxPool2d,
    DecomposedLayerNorm,
    DecomposedMultiHeadAttention,
    multiply_inverse_of_square_root
)
from torch.nn import *

torch.fx.wrap("multiply_inverse_of_square_root") # fx.wrap should be at the top of every module

from .core.hook import hook_context, activation_stats_hook
from functools import partial

unary_module_placeholder = Hardtanh

def scale_relu(net):
    print("Scaling ReLU")
    net, _ = pattern_matching_transform(
        net,
        patterns = [(torch.relu,), (F.relu,), (ReLU,)],
        graph_transform = replace_op(
            partial(
                ScaledOp,
                op = unary_module_placeholder,
                scale_transform = lambda x : x
            ),
            inherit_args = {'statistics':"activation_stats"}
        ),
        inplace = False
    )

    net, _ = pattern_matching_transform(
        net,
        patterns = [(unary_module_placeholder,)],
        graph_transform = replace_op(ReLU),
        inplace = False
    )
    return net

def collect_activations(net, loader, n_iter):
    print("collecting activations!")
    with hook_context(hook_fn = activation_stats_hook) as context:
        dev = next(net.parameters()).device
        for index, (x, _) in tqdm(enumerate(loader), desc = 'Collect Activations', total = n_iter):
            x = x.to(device=dev)
            net(x)
            if index > n_iter:
                break

@torch.no_grad()
def porting(net, loader, n_iter, relu_scaling):
    assert isinstance(relu_scaling, bool)

    net = _fuse_conv_bn(net)
    net = _decompose_maxpool(net)
    net = _decompose_layer_norm(net)
    net = _modularize_relu(net)
    net = _decompose_multihead_attention(net)

    if relu_scaling is not None:
        collect_activations(net, loader, n_iter)
        net = scale_relu(net)
    return net

def _fuse_conv_bn(model):
    print("Fuse conv/bn")
    model, _ = pattern_matching_transform(
        model,
        patterns = [
            (Conv1d, BatchNorm1d),
            (Conv2d, BatchNorm2d),
            (Conv3d, BatchNorm3d)
        ],
        graph_transform = fuse_conv_bn(),
        inplace = False
    )
    return model

def _decompose_maxpool(model):
    model, _ = pattern_matching_transform(
        model,
        patterns = [(F.max_pool2d,), (MaxPool2d,)],
        graph_transform = replace_op(
            BinaryTreeMaxPool2d,
            inherit_args = {
                'kernel_size':"kernel_size",
                "stride": "stride",
                "padding":"padding",
                "dilation" : "dilation"
            }
        ),
        inplace = False
    )
    return model

def _decompose_layer_norm(model):
    model, _ = pattern_matching_transform(
        model,
        patterns = [(LayerNorm,), (F.layer_norm,)],
        graph_transform = replace_op(
            DecomposedLayerNorm,
            inherit_args = {
                'normalized_shape':'normalized_shape',
                'eps':'eps',
                #'elementwise_affine':'elementwise_affine',
                'weight':'weight',
                'bias':'bias',
            }
        ),
        inplace = False
    )
    return model

# MHA not in article.
def _decompose_multihead_attention(model):
    model, _ = pattern_matching_transform(
        model,
        patterns = [(MultiheadAttention,)],
        graph_transform = replace_op(
            DecomposedMultiHeadAttention,
            inherit_args = {
                'embed_dim': "embed_dim",
                'num_heads':'num_heads',
                'dropout':'dropout',
                'add_zero_attn':'add_zero_attn',
                'q_proj_weight':'q_proj_weight',
                'k_proj_weight':'k_proj_weight',
                'v_proj_weight':'v_proj_weight',
                'in_proj_weight':'in_proj_weight',
                'out_proj':'out_proj',
                'in_proj_bias':'in_proj_bias',
                'bias_k':'bias_k',
                'bias_v':'bias_v',
                '_qkv_same_embed_dim':'_qkv_same_embed_dim',
                'batch_first':'batch_first',
            }
        ),
        inplace = False
    )
    return model


class Square(Module):
    def forward(self, input):
        return torch.square(input)
class Exp(Module):
    def forward(self, input):
        return torch.exp(input)
class Abs(Module):
    def forward(self, input):
        return torch.abs(input)
class Div(Module):
    def forward(self, input, other):
        return torch.div(input, other)
class Modular(nn.Module):
    def __init__(self, forward_fn):
        super().__init__()
        self.forward_fn = forward_fn
    def forward(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)

class ReLUWrapper(nn.Module):
    def __init__(self):
        super(ReLUWrapper, self).__init__()
        self.fn = nn.ReLU()
    def forward(self, input, *args, **kwargs):
        return self.fn(input)

def _modularize_relu(model):
    model, _ = pattern_matching_transform(
        model,
        patterns = [(torch.relu,), (F.relu,), (nn.ReLU,)],
        graph_transform = replace_op(
            unary_module_placeholder
        ),
        inplace = False
    )
    model, _ = pattern_matching_transform(
        model,
        patterns = [(unary_module_placeholder,)],
        graph_transform = replace_op(
            #nn.ReLU
            ReLUWrapper
        ),
        inplace = False
    )
    return model
