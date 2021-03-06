"""
    Model Name:

        AlexNet - using the Functional Keras API

        Replicated from the Caffe Zoo Model Version.

    Paper:

         ImageNet classification with deep convolutional neural networks by Krizhevsky et al. in NIPS 2012

    Alternative Example:

        Available at: http://caffe.berkeleyvision.org/model_zoo.html

        https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet

    Original Dataset:

        ILSVRC 2012

"""
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras import regularizers
from KerasLayers.Custom_layers import LRN2D

# global constants
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
LRN2D_norm = True       # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'tf'


def conv2D_lrn2d(x, nb_filter, nb_row, nb_col,
                 border_mode='same', subsample=(1, 1),
                 activation='relu', LRN2D_norm=True,
                 weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''

        Info:
            Function taken from the Inceptionv3.py script keras github


            Utility function to apply to a tensor a module Convolution + lrn2d
            with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Conv2D(nb_filter, (nb_row, nb_col),
               strides=subsample,
               activation=activation,
               padding=border_mode,
               kernel_regularizer=W_regularizer,
               activity_regularizer=b_regularizer,
               use_bias=False,
               data_format='channels_last')(x)
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)

    if LRN2D_norm:

        x = LRN2D(alpha=ALPHA, beta=BETA)(x)
        x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)

    return x


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

    # Channel 1 - Convolution Net Layer 1
    x = conv2D_lrn2d(img_input, 3, 11, 11, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), data_format='channels_last')(x)
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)

    # Channel 1 - Convolution Net Layer 2
    x = conv2D_lrn2d(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last')(x)
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)

    # Channel 1 - Convolution Net Layer 3
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x =  MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last')(x)
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)

    # Channel 1 - Convolution Net Layer 4
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)

    # Channel 1 - Convolution Net Layer 5
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)

    # Channel 1 - Cov Net Layer 6
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last')(x)
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)

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

'''
def check_print():
    # Create the Model
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[x])
    model.summary()

    # Save a PNG of the Model Build
    plot(model, to_file='./Model/AlexNet.png')

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print('Model Compiled')
'''
