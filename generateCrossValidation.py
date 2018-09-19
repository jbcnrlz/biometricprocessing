import os, sys, shutil, random, numpy as np, tensorflow as tf
from subprocess import check_output
from helper.functions import getFilesInPath
from PIL import Image as im
from networks.alexnet import *
from AlexNet import *
from keras.models import Model
from keras.utils import to_categorical

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def generateData(pathFiles):
    returnDataImages = []
    returnDataClass = []
    filesOnPath = getFilesInPath(pathFiles)
    for f in filesOnPath:
        #image = im.open(f)
        #image.thumbnail((224,224), im.ANTIALIAS)
        if f[-3:] == 'png':
            returnDataImages.append(f)
            classNumber = f.split(os.path.sep)[-1]
            classNumber = classNumber.split('_')[0]
            returnDataClass.append(int(classNumber))

    return returnDataImages, returnDataClass

def generateImageData(paths):
    returningPaths = []
    for p in paths:
        ni = im.open(p)
        returningPaths.append(np.array(ni))
    return np.array(returningPaths)

def getBaseGallery(dataFaces,classesFaces):
    returnBaseGallery = {}
    restOfData = []
    classesRestOfData = []
    for d in range(len(classesFaces)):
        currClass = classesFaces[d]
        if not currClass in returnBaseGallery.keys():
            returnBaseGallery[currClass] = dataFaces[d]
        else:
            classesRestOfData.append(currClass)
            restOfData.append(dataFaces[d])

    return list(returnBaseGallery.values()), list(returnBaseGallery.keys()), restOfData, classesRestOfData

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_to_tf(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))

def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = 'frgc_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)

def write_label_file(labels_to_class_names, dataset_dir,filename):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))

if __name__ == '__main__':
    imageData, classesData = generateData('generated_images_lbp_frgc')
    gBase, cgBase, rData, crData = getBaseGallery(imageData,classesData)
    foldSize = int(len(rData) / 10)
    foldResult = []

    foldChoices = random.sample([i for i in range(len(rData))], foldSize)
    foldProbe = []
    foldProbeClasses = []
    foldGallery = []
    foldGalleryClasses = []
    for i in range(len(rData)):
        if i in foldChoices:
            foldProbe.append(rData[i])
            foldProbeClasses.append(crData[i])
        else:
            foldGallery.append(rData[i])
            foldGalleryClasses.append(crData[i])

    foldGallery = generateImageData(gBase + foldGallery)
    foldGalleryClasses = np.array(cgBase + foldGalleryClasses)

    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(inputs=img_input,outputs=x)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                metrics=['accuracy'])

    checkpoint_path = "training/face_alexnet-{epoch:04d}.ckpt"

    if os.path.exists('training'):
        shutil.rmtree('training')

    os.makedirs('training')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=1)

    #y_binary = to_categorical(foldGalleryClasses - 1,num_classes=466)
    #model.fit(foldGallery,y_binary,epochs=10,batch_size=32,callbacks = [cp_callback])

    y_binary = to_categorical(np.array(foldProbeClasses) -1,num_classes=466)
    foldProbe = np.array(generateImageData(foldProbe))
    model.fit(foldProbe,y_binary,epochs=10,batch_size=32,callbacks = [cp_callback])

    '''

    y_binary = to_categorical(np.array(foldProbeClasses) -1,num_classes=224)
    foldProbe = np.array(generateImageData(foldProbe))
    a, acc = model.evaluate(foldProbe,y_binary)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    print('\n======================\n')
    for i in range(1,6):
        model.load_weights('training/face_alexnet-000'+str(i)+'.ckpt')
        a, acc = model.evaluate(np.array(foldProbe),y_binary)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    '''

    '''
    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            output_filename = _get_dataset_filename('generated_images_lbp_frgc','validation', 0,1)
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in range(len(foldProbe)):
                    image_data = tf.gfile.FastGFile(foldProbe[i], 'rb').read()
                    height, width = image_reader.read_image_dims(sess, image_data)
                    class_id = foldProbeClasses[i]
                    faceTf = image_to_tf(image_data, b'png', height, width, class_id)
                    tfrecord_writer.write(faceTf.SerializeToString())

            output_filename = _get_dataset_filename('generated_images_lbp_frgc', 'train', 0, 1)
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in range(len(foldGallery)):
                    image_data = tf.gfile.FastGFile(foldGallery[i], 'rb').read()
                    height, width = image_reader.read_image_dims(sess, image_data)
                    class_id = foldGalleryClasses[i]
                    faceTf = image_to_tf(image_data, b'png', height, width, class_id)
                    tfrecord_writer.write(faceTf.SerializeToString())

    uniqueClassesGallery = set(sorted(foldGalleryClasses))
    labels_to_class_names = dict(zip(range(len(uniqueClassesGallery)), uniqueClassesGallery))
    write_label_file(labels_to_class_names, 'generated_images_lbp_frgc','labels.txt')

    print('oi')
    '''