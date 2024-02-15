import os
import shutil
import tensorflow as tf
from models import build_models
from losses import build_losses
from optimizers import build_optimizer
from metrics import build_metrics
from callbacks import build_callbacks
from data_utils.data_flow import get_train_test_data
from utils.train_processing import create_folder_weights, train_prepare
from utils.config_processing import load_config


def train(file_config=None):

    config = load_config(file_config)
    train_config = config['Train']
    data_config  = config['Dataset']
              
    if train_prepare(train_config['mode']):
        TRAINING_TIME_PATH = create_folder_weights(train_config['save_weight_path'])
        shutil.copy(file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(file_config)))
        
        converter, model = build_models(config['Model'])
        train_generator, valid_generator, test_generator = get_train_test_data(data_dirs           = data_config['data_dir'],
                                                                               annotation_dirs     = data_config['annotation_dir'],
                                                                               target_size         = config['Model']['input_shape'], 
                                                                               batch_size          = train_config['batch_size'],
                                                                               character           = data_config['data_info']['character'],
                                                                               character_converter = converter,
                                                                               max_string_length   = data_config['data_info']['max_string_length'], 
                                                                               sensitive           = data_config['data_info']['sensitive'],
                                                                               color_space         = data_config['data_info']['color_space'],
                                                                               augmentor           = data_config['data_augmentation'],
                                                                               normalizer          = data_config['data_normalizer']['norm_type'],
                                                                               mean_norm           = data_config['data_normalizer']['norm_mean'],
                                                                               std_norm            = data_config['data_normalizer']['norm_std'],
                                                                               resize_with_pad     = data_config['data_normalizer']['norm_resize_with_pad'],
                                                                               data_type           = data_config['data_info']['data_type'],
                                                                               check_data          = data_config['data_info']['check_data'],
                                                                               load_memory         = data_config['data_info']['load_memory'],
                                                                               dataloader_mode     = data_config['data_loader_mode'])

        optimizer = build_optimizer(config['Optimizer'])
        losses    = build_losses(config['Losses'])
        metrics   = build_metrics(config['Metrics'])
        callbacks = build_callbacks(config['Callbacks'], TRAINING_TIME_PATH)
        
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        
        if valid_generator is not None:
            model.fit(train_generator,
                      steps_per_epoch  = train_generator.N // train_config['batch_size'],
                      validation_data  = valid_generator,
                      validation_steps = valid_generator.N // train_config['batch_size'],
                      epochs           = train_config['epoch']['end'],
                      initial_epoch    = train_config['epoch']['start'],
                      callbacks        = callbacks)
        else:
            model.fit(train_generator,
                      steps_per_epoch     = train_generator.n // train_config['batch_size'],
                      epochs              = train_config['epoch']['end'],
                      initial_epoch       = train_config['epoch']['start'],
                      callbacks           = callbacks)
            
        if test_generator is not None:
            model.evaluate(test_generator)
            
        model.save_weights(TRAINING_TIME_PATH + 'weights/last_weights', save_format=train_config['save_weight_type'])
        
        
if __name__ == '__main__':
    train('./configs/abinet/tps-abinet.yaml')
