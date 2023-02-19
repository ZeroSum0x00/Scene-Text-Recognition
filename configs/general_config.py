# STR hyper-parameters
STR_FILTERS                     = [64, 128, 256, 512]

STR_HIDDEN_DIMENTION            = 256

STR_OUTPUT_DIMENTION            = 256

STR_TARGET_SIZE                 = [32, 585, 1]


# Training hyper-parameters
DATA_PATH                       = "/home/vbpo/Desktop/TuNIT/working/Datasets/IIIT5K"

DATA_ANNOTATION_PATH            = "/home/vbpo/Desktop/TuNIT/working/Datasets/IIIT5K/annotations"

DATA_DESTINATION_PATH           = None

DATA_AUGMENTATION               = None

DATA_NORMALIZER                 = 'sub_divide'

DATA_CHARACTER                  = "0123456789abcdefghijklmnopqrstuvwxyz"

DATA_MAX_STRING_LENGTH          = 25

DATA_SENSITIVE                  = False

DATA_TYPE                       = 'json'

CHECK_DATA                      = False

DATA_LOAD_MEMORY                = False

TRAIN_BATCH_SIZE                = 32

TRAIN_EPOCH_INIT                = 0

TRAIN_EPOCH_END                 = 1000

TRAIN_OPTIMIZER                 = 'adam'

TRAIN_LR                        = 0.001

TRAIN_WEIGHT_TYPE               = None

TRAIN_WEIGHT_OBJECTS            = [        
                                    {
                                      'path': './saved_weights/20230125-083355/best_weights',
                                      'stage': 'full',
                                      'custom_objects': None
                                    }
                                  ]

TRAIN_RESULT_SHOW_FREQUENCY     = 10

TRAIN_SAVE_WEIGHT_FREQUENCY     = 100

TRAIN_SAVED_PATH                = './saved_weights/'

TRAIN_MODE                      = 'graph'
