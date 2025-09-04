import torch
import torch.nn.functional as F

from itertools import islice
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .graph_functional import pattern_matching_transform
from .graph_functional.utils import replace_node_module
from .graph_functional.transforms import replace_op, replace_ops_cases

from .core.layer import (
    BinaryTreeMaxPool2d,
    DecomposedLayerNorm,
    DecomposedMultiHeadAttention,
    multiply_inverse_of_square_root
)
from torch.nn import *
from torch.nn.utils.fusion import fuse_conv_bn_eval

# fx.wrap should be at the top of every module
torch.fx.wrap("multiply_inverse_of_square_root")

from .core.hook import hook_context, activation_stats_hook
from functools import partial

# class SignPreservingScaler(Module):
#     def __init__(self, scale, inverse):
#         super().__init__()
#         self.inverse = inverse
#         self.scale_factor = scale

#     def forward(self,x):
#         if not self.inverse:
#             x *= self.scale_factor
#         else:
#             x /= self.scale_factor
#         return x

class ScaledOp(Module):
    def __init__(self, scale_transform, statistics):
        super().__init__()

        print("STATS", statistics["output/max"].shape)

        self.scale = statistics["output/max"]

        self.scale[self.scale <= 1e-5] = 1.0 # Prevent nan

        self.scale = 1.0 / torch.unsqueeze(torch.abs(self.scale), dim = 0)

        # self.forward_scaler = SignPreservingScaler(
        #     scale = self.scale, inverse = False)
        self.relu = Hardtanh()
        # self.backward_scaler = SignPreservingScaler(
        #     scale = scale_transform(self.scale),
        #     inverse = True
        # )
    def forward(self,x):
        print("Scaled op, forward")

        y = x * self.scale
        y = self.relu(y)
        y = y  / self.scale


        # y = self.forward_scaler(x)
        # y = self.relu(y)
        # y = self.backward_scaler(y)
        return y

#torch.fx.wrap("ScaledOp")

unary_module_placeholder = Hardtanh

def scale_relu(net):
    print("Scaling ReLU")
    net, _ = pattern_matching_transform(
        net,
        patterns = [(torch.relu,), (F.relu,), (ReLU,)],
        graph_transform = replace_op(
            partial(
                ScaledOp,
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
    net = decompose_layer_norm(net)
    net = modularize_relu(net)
    net = _decompose_multihead_attention(net)

    if relu_scaling is not None:
        collect_activations(net, loader, n_iter)
        net = scale_relu(net)
    return net

def fuse_conv_bn_node_transform(node, trail, pattern, graph, modules):
    assert len(pattern) >= 2
    assert len(trail) >= 2

    node_prev = trail[-2]

    # Output of conv is used by other nodes
    if len(node_prev.users) > 1:
        return

    conv = modules[node_prev.target]
    bn = modules[node.target]
    fused_conv = fuse_conv_bn_eval(conv, bn)

    replace_node_module(node_prev, modules, fused_conv)
    node.replace_all_uses_with(node_prev)
    graph.erase_node(node)
    return



# def fuse_conv_bn():
#     def fuse_conv_bn_node_transform(
#             node, trail, pattern, graph, modules
#     ):
#         assert len(pattern) >= 2
#         assert len(trail) >= 2

#         node_prev = trail[-2]

#         # Output of conv is used by other nodes
#         if len(node_prev.users) > 1:
#             return

#         conv = modules[node_prev.target]
#         bn = modules[node.target]
#         fused_conv = fuse_conv_bn_eval(conv, bn)

#         replace_node_module(node_prev, modules, fused_conv)
#         node.replace_all_uses_with(node_prev)
#         graph.erase_node(node)
#         return

#     return fuse_conv_bn_node_transform

def _fuse_conv_bn(net):
    print("Fuse conv/bn")
    net, _ = pattern_matching_transform(
        net,
        patterns = [
            (Conv1d, BatchNorm1d),
            (Conv2d, BatchNorm2d),
            (Conv3d, BatchNorm3d)
        ],
        graph_transform = fuse_conv_bn_node_transform,
        inplace = False
    )
    return net

def _decompose_maxpool(net):
    net, _ = pattern_matching_transform(
        net,
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
    return net

def decompose_layer_norm(net):
    net, _ = pattern_matching_transform(
        net,
        patterns = [(LayerNorm,), (F.layer_norm,)],
        graph_transform = replace_op(
            DecomposedLayerNorm,
            inherit_args = {
                'normalized_shape':'normalized_shape',
                'eps':'eps',
                'weight':'weight',
                'bias':'bias',
            }
        ),
        inplace = False
    )
    return net

# MHA not in article.
def _decompose_multihead_attention(net):
    net, _ = pattern_matching_transform(
        net,
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
    return net


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

# Why need this?
class ReLUWrapper(Module):
    def __init__(self):
        super(ReLUWrapper, self).__init__()
        self.fn = ReLU()
    def forward(self, input, *args, **kwargs):
        return self.fn(input)

def modularize_relu(net):
    net, _ = pattern_matching_transform(
        net,
        patterns = [(torch.relu,), (F.relu,), (ReLU,)],
        graph_transform = replace_op(
            unary_module_placeholder
        ),
        inplace = False
    )
    net, _ = pattern_matching_transform(
        net,
        patterns = [(unary_module_placeholder,)],
        graph_transform = replace_op(
            ReLUWrapper
        ),
        inplace = False
    )
    return net
