import copy
import importlib
from .scene_text_recognition import STR
from .crnn import CRNN
from .acrnn import ACRNN
from .abinet import ABINet
from .satrn import SATRN
from .layers import (CTCLabelConverter, OnehotLabelConverter, SparseOnehotLabelConverter, AttnLabelConverter,
                     TPS_SpatialTransformerNetwork, TPS_SpatialTransformerNetworkV2, SimpleSpatialTransformer,
                     STRAttention,
                     GRU, LSTM, BidirectionalLSTM, CascadeBidirectionalLSTM, MDLSTM, ConvolutionHead, SimpleSVTRHead, EncodeSVTRHead)
from .architectures import (ShallowCNN, VGG_FeatureExtractor, GRCNN_FeatureExtractor, ResNet_FeatureExtractor, 
                            HRNet_FeatureExtractor, ResNet34, LCNet, SVTRNet)


def build_models(config, weights=None):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    input_shape = config.pop("input_shape")
    weight_path = config.pop("weight_path")
    load_weight_type = config.pop("load_weight_type")

    converter_config = config['LabelConverter']
    converter_name = converter_config.pop("name")
    converter = getattr(mod, converter_name)(**converter_config)
    num_class = converter.N

    architecture_config = config['Architecture']
    architecture_name = architecture_config.pop("name")
    
    backbone_config = config['Backbone']
    backbone_config['input_shape'] = input_shape
    if 'out_channels' in backbone_config:
        backbone_config['classes'] = backbone_config['out_channels']
        del backbone_config['out_channels']
    backbone_name = backbone_config.pop("name")
    backbone = getattr(mod, backbone_name)(**backbone_config)
    architecture_config['backbone'] = backbone
    
    if architecture_name.lower() == "crnn" or architecture_name.lower() == "acrnn":

        if 'SequenceLayer' in config and config['SequenceLayer']:
            sequen_layer_config = config['SequenceLayer']
            sequen_layer_name = sequen_layer_config.pop("name")
            if sequen_layer_name == "ConvolutionHead" or sequen_layer_name == "SVTRHead":
                sequen_layer_config['num_classes'] = num_class
            sequen_layer = getattr(mod, sequen_layer_name)(**sequen_layer_config)
        else:
            sequen_layer = None
    

        if architecture_name.lower() == "acrnn":
            if 'AttentionLayer' in config and config['AttentionLayer']:
                attn_layer_config = config['AttentionLayer']
                attn_layer_config['num_classes'] = num_class
                attn_layer_name = attn_layer_config.pop("name")
                attn_layer = getattr(mod, attn_layer_name)(**attn_layer_config)
            else:
                attn_layer = None
            
            architecture_config['attention_net'] = attn_layer
        else:
            architecture_config['num_classes'] = num_class

        architecture_config['sequence_net'] = sequen_layer

    if 'TransformLayer' in config and config['TransformLayer']:
        transform_layer_config = config['TransformLayer']
        transform_layer_name = transform_layer_config.pop("name")
        transform_layer = getattr(mod, transform_layer_name)(**transform_layer_config)
    else:
        transform_layer = None

    if architecture_name.lower() == 'abinet':
        architecture_config['num_classes'] = num_class
        
    architecture_config['transform_net'] = transform_layer

    architecture = getattr(mod, architecture_name)(**architecture_config)
    model = STR(architecture)

    if weights:
        model.load_weights(weights)
    else:
        if load_weight_type and weight_path:
            if load_weight_type == "weights":
                model.load_weights(weight_path)
            elif load_weight_type == "models":
                model.load_models(weight_path)
    return converter, model