import tensorflow as tf
from pathlib import Path
import os
import matplotlib.pyplot as plt
import glob


def get_cat_dirs(base_dir, sort_by=None):
    """
    Return a list with cat directories.
    Args:
        base_dir: baseline dir for images
        sort_by: function to sort the cats by
    """

    return  sorted(glob.glob(base_dir + '/*'), key=sort_by)


def get_cat_image_paths(cat_paths=None, type='raw', base_dir=None, sort_by=None):
    """
    Return a list with cats images from a list of cats paths.
    Args:
        cat_paths: list with cats directories paths
        type: which type of images to extract. Available types are 'raw' or 'face', defaults to 'raw
        base_dir: base images dir, required only if cat_paths is not provided
        sort_by: function by which to sort the cats
    """

    # cat cat dirs if not provided
    if cat_paths is None:
        base_dir = base_dir + '/images'
        cat_paths = get_cat_dirs(base_dir, sort_by=sort_by)

    images_paths = []

    for cat_dir in cat_paths:
        cat_images = glob.glob(cat_dir + '/' + type + '/*')
        images_paths += cat_images

    return images_paths


def read_image(path, return_cat_class=True):
    """Read jpeg image from tensor path"""
    image = tf.io.decode_jpeg(tf.io.read_file(path))

    if return_cat_class:
        cat_class = tf.strings.split(path, sep='/')[-3]
        return cat_class, image

    else:
        return image


def resize_image(image, height=256, width=256):
    """Resize tensor jpeg image to height and width"""
    return tf.cast(tf.image.resize_with_pad(image, height, width), tf.uint8)


def image_generator(images_dir, image_size=(256, 256), type='raw', sort_by=None):
    """Return dataset image generator. Each image will be resized to image_size.
        Args:
            image_size: target size for images. aspect ration will be reserved
            type: which type of images to extract. Available types are 'raw' or 'face', defaults to 'raw
            sort_by: function by which to sort the cats
    """

    # get cat images paths dataset
    image_paths = get_cat_image_paths(type=type, base_dir=images_dir, sort_by=sort_by)
    image_paths_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # read and resize image to image_size
    image_dataset = image_paths_dataset.map(read_image)

    # helper function
    def resize_func(cat_class, image):
        return cat_class, resize_image(image, height=image_size[0], width=image_size[1])

    image_dataset = image_dataset.map(resize_func)

    return image_dataset


if __name__ == '__main__':
    images_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ds = image_generator(images_dir)
    for cat_class, image in ds.batch(10):
        print(cat_class.shape, image.shape)
