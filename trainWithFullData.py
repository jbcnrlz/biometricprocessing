import os, tensorflow as tf, numpy as np
from generateCrossValidation import *
from networks.small_AlexNet import *
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('-n', '--network', default=None, help='Name for the model', required=False)
    parser.add_argument('-p', '--pathBase',default='generated_images_lbp_frgc',help='Path for faces', required=False)
    parser.add_argument('-b', '--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('-c', '--classNumber', type=int, default=466, help='Quantity of classes', required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to be run', required=False)
    args = parser.parse_args()

    if not os.path.exists('training'):
        os.makedirs('training')

    if not os.path.exists(os.path.join('training', 'full')):
        os.makedirs(os.path.join('training', 'full'))

    checkpoint_path = None if args.network is None else "training/full/face_"+args.network+"-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,period=args.epochs)
    imageData, classesData = generateData(args.pathBase)
    imageData = np.array(generateImageData(imageData))
    classesData = np.array(classesData)

    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model(numClasses=args.classNumber)

    model = Model(inputs=img_input, outputs=x)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(
        generateDataFromArray(imageData,classesData, args.batch,args.classNumber),
        steps_per_epoch=math.ceil(imageData.shape[0] / args.batch),
        verbose=1,
        epochs=args.epochs,
        callbacks=[cp_callback]
    )