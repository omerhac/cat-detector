import tensorflow as tf
from triplet_loss import *


def get_model(input_shape=(256, 256)):
    """Return a cat verificator model"""

    # initiate layers
    inp_image = tf.keras.layers.Input(input_shape)
    efnet = tf.keras.applications.EfficientNetB2(include_top=False, pooling='avg')
    dense_rep = tf.keras.layers.Dense(64, activation='linear')

    model = tf.keras.models.Sequential([
        inp_image,
        efnet,
        dense_rep
    ])

    return model


if __name__ == '__main__':
    get_model()