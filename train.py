import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from models import STR, CRNN, VGG, CTCLabelConverter
from losses import CTCLoss
from metrics import CTCAccuracy
from callbacks import LossHistory, AccuracyHistory
from data_utils.data_flow import get_train_test_data
from utils.train_processing import create_folder_weights, train_prepare
from configs import general_config as cfg


def train(data_path                   = cfg.DATA_PATH,
          data_anno_path              = cfg.DATA_ANNOTATION_PATH,
          data_dst_path               = cfg.DATA_DESTINATION_PATH,
          data_normalizer             = cfg.DATA_NORMALIZER,
          data_augmentation           = cfg.DATA_AUGMENTATION,
          character                   = cfg.DATA_CHARACTER, 
          max_string_length           = cfg.DATA_MAX_STRING_LENGTH, 
          sensitive                   = cfg.DATA_SENSITIVE, 
          data_type                   = cfg.DATA_TYPE,
          check_data                  = cfg.CHECK_DATA,
          load_memory                 = cfg.DATA_LOAD_MEMORY,
          str_filters                 = cfg.STR_FILTERS,
          str_hidden_dim              = cfg.STR_HIDDEN_DIMENTION,
          str_output_dim              = cfg.STR_OUTPUT_DIMENTION,
          input_shape                 = cfg.STR_TARGET_SIZE,
          batch_size                  = cfg.TRAIN_BATCH_SIZE,
          init_epoch                  = cfg.TRAIN_EPOCH_INIT,
          end_epoch                   = cfg.TRAIN_EPOCH_END,
          lr_init                     = cfg.TRAIN_LR, 
          weight_type                 = cfg.TRAIN_WEIGHT_TYPE,
          weight_objects              = cfg.TRAIN_WEIGHT_OBJECTS,
          show_frequency              = cfg.TRAIN_RESULT_SHOW_FREQUENCY,
          saved_weight_frequency      = cfg.TRAIN_SAVE_WEIGHT_FREQUENCY,
          saved_path                  = cfg.TRAIN_SAVED_PATH,
          training_mode               = cfg.TRAIN_MODE):
    if train_prepare(training_mode):
        TRAINING_TIME_PATH = create_folder_weights(saved_path)
        
        train_generator, valid_generator = get_train_test_data(data_zipfile      = data_path,
                                                               annotation_dir    = data_anno_path,
                                                               dst_dir           = data_dst_path, 
                                                               target_size       = input_shape, 
                                                               batch_size        = batch_size,
                                                               character         = character,
                                                               max_string_length = max_string_length, 
                                                               sensitive         = sensitive,
                                                               augmentor         = data_augmentation,
                                                               normalizer        = data_normalizer,
                                                               data_type         = data_type,
                                                               check_data        = check_data,
                                                               load_memory       = load_memory)

        converter = CTCLabelConverter(character)
        num_class = converter.N

        backbone = VGG(str_filters)
        architecture = CRNN(backbone, str_filters, str_hidden_dim, num_class)
        model = STR(architecture)

        if weight_type and weight_objects:
            if weight_type == "weights":
                model.load_weights(weight_objects)
            elif weight_type == "models":
                model.load_models(weight_objects)

        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_delta=0, verbose=1)
        
        optimizer = Adam(learning_rate=0.0001, global_clipnorm=5.0)
        
        losses = [
            {'loss': CTCLoss(), 'coeff': 1},
        ]
        
        metric_accuracy = CTCAccuracy()
        
        loss_history = LossHistory(result_path=TRAINING_TIME_PATH)
        
        accuracy_history = AccuracyHistory(result_path=TRAINING_TIME_PATH, save_best=True)
        
        checkpoint = ModelCheckpoint(TRAINING_TIME_PATH + 'checkpoint_{epoch:04d}/saved_str_weights', 
                                     monitor='val_loss',
                                     verbose=1, 
                                     save_weights_only=True,
                                     save_freq="epoch",
                                     period=saved_weight_frequency)
        
        logger = CSVLogger(TRAINING_TIME_PATH + 'train_history.csv', separator=",", append=True)
        
        callbacks = [accuracy_history, loss_history, checkpoint, logger]

        model.compile(optimizer=optimizer, loss=losses, metrics=[metric_accuracy])
        
        history = model.fit(train_generator,
                            steps_per_epoch  = train_generator.N // batch_size,
                            validation_data  = valid_generator,
                            validation_steps = valid_generator.N // batch_size,
                            epochs           = end_epoch,
                            initial_epoch    = init_epoch,
                            callbacks        = callbacks)
        
        model.save_weights(TRAINING_TIME_PATH + 'best_weights', save_format="tf")
        
        
if __name__ == '__main__':
    train()
