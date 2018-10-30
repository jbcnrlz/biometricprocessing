from keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D, Dropout

def base_model(imShape=(100,100,4),numClasses=466):
    img_input = Input(shape=imShape)

    model = Conv2D(64, 3, strides=1, padding='same', input_shape=imShape, name='conv1_1')(img_input)
    model = Activation('relu')(model)

    model = Conv2D(64, 3, strides=1, padding='same', input_shape=imShape, name='conv1_2')(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(2,2)(model)

    model = Conv2D(128, 3, strides=1, padding='same', input_shape=imShape, name='conv2_1')(model)
    model = Activation('relu')(model)

    model = Conv2D(128, 3, strides=1, padding='same', input_shape=imShape, name='conv2_2')(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(2,strides=2)(model)

    model = Conv2D(256, 3, strides=1, padding='same', input_shape=imShape, name='conv3_1')(model)
    model = Activation('relu')(model)

    model = Conv2D(256, 3, strides=1, padding='same', input_shape=imShape, name='conv3_2')(model)
    model = Activation('relu')(model)

    model = Conv2D(256, 3, strides=1, padding='same', input_shape=imShape, name='conv3_3')(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(2, strides=2)(model)

    model = Conv2D(512, 3, strides=1, padding='same', input_shape=imShape, name='conv4_1')(model)
    model = Activation('relu')(model)

    model = Conv2D(512, 3, strides=1, padding='same', input_shape=imShape, name='conv4_2')(model)
    model = Activation('relu')(model)

    model = Conv2D(512, 3, strides=1, padding='same', input_shape=imShape, name='conv4_3')(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(2, strides=2)(model)

    model = Conv2D(512, 3, strides=1, padding='same', input_shape=imShape, name='conv5_1')(model)
    model = Activation('relu')(model)

    model = Conv2D(512, 3, strides=1, padding='same', input_shape=imShape, name='conv5_2')(model)
    model = Activation('relu')(model)

    model = Conv2D(512, 3, strides=1, padding='same', input_shape=imShape, name='conv5_3')(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(2, strides=2)(model)

    model = Flatten()(model)
    model = Dense(4096, activation='relu')(model)
    model = Dropout(0.5)(model)

    model = Dense(4096, activation='relu')(model)
    model = Dropout(0.5)(model)

    model = Dense(numClasses, activation='softmax')(model)

    return model, img_input