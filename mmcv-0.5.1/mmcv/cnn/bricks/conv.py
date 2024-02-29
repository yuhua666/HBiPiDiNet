from torch import nn as nn
from .BiConv import BiConv, Conv2d_cd, BiConv2d_cd, BiCDC, BiConv2d_cs, BiConv2d_csenc, BiConv2d_pool
from .BiConv import BiConv_LBP, LBPConv, LBPConv2, LBPConv3

from .registry import CONV_LAYERS

CONV_LAYERS.register_module('Conv1d', module=nn.Conv1d)
CONV_LAYERS.register_module('Conv2d', module=nn.Conv2d)
CONV_LAYERS.register_module('Conv3d', module=nn.Conv3d)
CONV_LAYERS.register_module('Conv', module=nn.Conv2d)
CONV_LAYERS.register_module('BiConv', module=BiConv)
CONV_LAYERS.register_module('Conv2d_cd', module=Conv2d_cd)
CONV_LAYERS.register_module('BiConv2d_cd', module=BiConv2d_cd)
CONV_LAYERS.register_module('BiConv2d_cs', module=BiConv2d_cs)
CONV_LAYERS.register_module('BiConv2d_csenc', module=BiConv2d_csenc)
CONV_LAYERS.register_module('BiCDC', module=BiCDC)
CONV_LAYERS.register_module('BiConv_LBP', module=BiConv_LBP)
CONV_LAYERS.register_module('BiConv2d_pool', module=BiConv2d_pool)
CONV_LAYERS.register_module('LBPConv', module=LBPConv)
CONV_LAYERS.register_module('LBPConv2', module=LBPConv2)
CONV_LAYERS.register_module('LBPConv3', module=LBPConv3)



def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
