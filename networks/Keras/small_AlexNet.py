from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras import regularizers
from KerasLayers.Custom_layers import LRN2D

# global constants
ALPHA = 0.0001
BETA = 0.75
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
LRN2D_norm = True       # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'tf'


def create_model(imShape=(100,100),channels=4,numClasses=466):
    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (channels, imShape[0], imShape[1])  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (imShape[0], imShape[1], channels)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))


    x = Conv2D(64,(8,8),strides=(4,4),activation='relu',padding='same',
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        activity_regularizer=regularizers.l2(WEIGHT_DECAY),
        use_bias=False,
        data_format='channels_last'
    )(img_input)

    # Channel 1 - Cov Net Layer 7
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Channel 1 - Cov Net Layer 8
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Final Channel - Cov Net 9
    x = Dense(numClasses,activation='softmax')(x)

    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING