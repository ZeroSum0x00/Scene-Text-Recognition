import copy
import importlib
import tensorflow as tf

from .ctc_loss import CTCLoss
from .attn_cross_entropy import AttentionEntropyLoss
from .cascade_cross_entropy import CascadeCrossentropy



def build_losses(config):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    losses = []
    for cfg in config:
        name = str(list(cfg.keys())[0])
        value = list(cfg.values())[0]
        coeff = value.pop("coeff")
        arch = getattr(mod, name)(**value)
        losses.append({'loss': arch, 'coeff': coeff})
    return losses