import copy
import importlib
from .ctc_character_accuracy import CTCCharacterAccuracy
from .ctc_word_accuracy import CTCWordAccuracy
from .entropy_character_accuracy import EntropyCharAccuracy
from .entropy_word_accuracy import EntropyWordAccuracy
from .attention_character_accuracy import AttentionCharAccuracy
from .attention_word_accuracy import AttentionWordAccuracy


def build_metrics(config):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    metrics = []
    if config:
        for cfg in config:
            name = str(list(cfg.keys())[0])
            value = list(cfg.values())[0]
            if not value:
                value = {}
            arch = getattr(mod, name)(**value)
            metrics.append(arch)
    return metrics