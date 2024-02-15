import copy
import importlib
from .geometric import *
from .photometric import *


def build_augmenter(config):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    augmenter = []

    for cfg in config:
        name = str(list(cfg.keys())[0])
        value = list(cfg.values())[0]

        if not value:
            value = {}
        arch = getattr(mod, name)(**value)
        augmenter.append(arch)
    return augmenter