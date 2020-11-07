import tensorflow as tf
from pathlib import Path
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    images_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'

    ds = tf.keras.preprocessing.image.ImageDataGenerator(fill_mode='constant',
                                                         cval=0).flow_from_directory(
        images_dir,
        class_mode='sparse',
        batch_size=1,
        target_size=(512, 512),
        interpolation='lanczos',
        save_to_dir=images_dir + '/check'
    )
    for image, c in ds:
        print(c, image.shape)