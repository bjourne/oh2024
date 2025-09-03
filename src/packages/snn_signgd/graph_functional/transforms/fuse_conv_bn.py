from torch import fx
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Callable, Any
from ..utils import replace_node_module
