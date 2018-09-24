from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np


def base_model(imShape=(100,100,4),numClasses=2622,weights_path=None,dFormat="channels_last"):
    img = Input(shape=imShape)

    x = ZeroPadding2D(padding=(1, 1))(img)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),data_format=dFormat)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),data_format=dFormat)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),data_format=dFormat)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),data_format=dFormat)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),data_format=dFormat)(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = Dense(numClasses, activation='softmax', name='fc8')(x)

    model = Model(inputs=img, outputs=x)

    if weights_path:
        model.load_weights(weights_path)

    return model

'''
if __name__ == "__main__":
    im = Image.open('pretrained_models/test.jpg')
    im = im.resize((224, 224))
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = vgg_face('pretrained_models/vgg-face-keras-fc.h5')
    out = model.predict(im)
    print(out[0][0])
'''