import torch

from torch import nn
from torch.nn import *

import torch.nn.functional as F

from .graph_functional import pattern_matching_transform
from .graph_functional.transforms import replace_op, replace_ops_cases
from spikingjelly.activation_based import functional

from typing import Callable, List

from tqdm import tqdm
from functools import partial
import copy

from .core.layer import multiply_inverse_of_square_root
# fx.wrap should be at the top of every module
#torch.fx.wrap("multiply_inverse_of_square_root")
