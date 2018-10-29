from keras import backend as K
import tensorflow as tf, functools

def _center_loss_func(features, labels, alpha, num_classes,centers, feature_dim):
    assert feature_dim == features.get_shape()[1]
    #labels = K.reshape(labels, [-1])
    labels = tf.argmax(labels, axis=1)
    labels = tf.to_int32(labels)
    centers_batch = tf.gather(centers, labels)
    diff = (1 - alpha) * (centers_batch - features)
    centers = tf.scatter_sub(centers, labels, diff)
    loss = tf.reduce_mean(K.square(features - centers_batch))
    return loss

def get_center_loss(alpha, num_classes, feature_dim):
    """Center loss based on the paper "A Discriminative
       Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    # Each output layer use one independed center: scope/centers
    centers = K.zeros([num_classes, feature_dim])
    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return _center_loss_func(y_pred, y_true, alpha,
                                 num_classes, centers, feature_dim)
    return center_loss