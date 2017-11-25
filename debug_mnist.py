"""A debugging sample for a broken keras program which tries to learns MNIST.

This is heavily based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/debug/examples/debug_mnist.py
and https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py .
The neural network in this demo has a problem with numerical computation (Infs and NaNs).
"""

import sys

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import TruncatedNormal, Constant
from keras.optimizers import Adam
from keras import backend as K


IMAGE_SIZE = (28, 28)
HIDDEN_SIZE = 500
NUM_LABELS = 10
RAND_SEED = 42

FLAT_IMAGE_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]


def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    K.set_session(sess)


def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], FLAT_IMAGE_SIZE)
    x_test = x_test.reshape(x_test.shape[0], FLAT_IMAGE_SIZE)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NUM_LABELS)
    y_test = keras.utils.to_categorical(y_test, NUM_LABELS)

    return x_train, y_train, x_test, y_test


def build_model():
    def reproducable_trunc_norm():
        return TruncatedNormal(mean=0.0, stddev=0.1, seed=RAND_SEED)

    def reproducable_const():
        return Constant(value=0.1)

    def unstable_categorical_crossentropy(y_true, y_pred):
        return - tf.reduce_sum(y_true * tf.log(y_pred), axis=len(y_pred.get_shape()) - 1)

    model = Sequential()
    model.add(Dense(
        HIDDEN_SIZE, input_shape=(FLAT_IMAGE_SIZE, ), activation='relu',
        kernel_initializer=reproducable_trunc_norm(), bias_initializer=reproducable_const()))
    model.add(Dense(
        NUM_LABELS, activation='softmax',
        kernel_initializer=reproducable_trunc_norm(), bias_initializer=reproducable_const()))

    model.compile(loss=unstable_categorical_crossentropy, optimizer=Adam(lr=0.025), metrics=['accuracy'])  # this line causes -inf!
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.025), metrics=['accuracy'])  # this is correct
    model.summary()
    return model


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--debug':
            set_debugger_session()
        else:
            raise ValueError('unkown option :{}'.format(sys.argv[1]))
    x_train, y_train, x_test, y_test = get_mnist_dataset()
    model = build_model()
    model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
