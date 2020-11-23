from abc import ABC

import tensorflow as tf
from triplet_loss import *


class CatVerificator(tf.keras.Model, ABC):
    """Cat verificator model"""

    def __init__(self, input_shape=(256, 256, 3)):
        super(CatVerificator, self).__init__()
        self._efnet = tf.keras.applications.EfficientNetB2(include_top=False, pooling='avg', input_shape=input_shape)
        self._dense_rep = tf.keras.layers.Dense(64, activation='linear', name='dense_rep')

    def __call__(self, x):
        efnet_out = self._efnet(x)
        dense_rep = self._dense_rep(efnet_out)

        # l2 normalize
        dense_rep = tf.keras.backend.l2_normalize(dense_rep)

        return dense_rep


if __name__ == '__main__':
    a = CatVerificator()
    b = tf.random.uniform((256, 256, 3))
    print(b.shape)
    print(a(b).shape)