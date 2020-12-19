import tensorflow as tf
import os
import imghdr
import glob
import numpy as np


def get_cat_dirs(base_dir, sort_by=None):
    """
    Return a list with cat directories.
    Args:
        base_dir: baseline dir for images
        sort_by: function to sort the cats by
    """

    return sorted(glob.glob(base_dir + '/*'), key=sort_by)


def get_cat_image_paths(cat_paths=None, type='raw', images_dir=None, sort_by=None):
    """
    Return a list with cats images from a list of cats paths.
    Args:
        cat_paths: list with cats directories paths
        type: which type of images to extract. Available types are 'raw' or 'face', defaults to 'raw
        images_dir: base images dir, required only if cat_paths is not provided
        sort_by: function by which to sort the cats
    """

    # get cat dirs if not provided
    if cat_paths is None:
        cat_paths = get_cat_dirs(images_dir, sort_by=sort_by)

    images_paths = []

    for cat_dir in cat_paths:
        cat_images = glob.glob(cat_dir + '/' + type + '/*.jpg')
        images_paths += cat_images

    return images_paths


def read_image(path, return_cat_class=True):
    """Read jpeg image from tensor path"""

    # handling corrupted images
    try:
        image = tf.io.decode_jpeg(tf.io.read_file(path))
    except:
        image = tf.zeros([256, 256, 3], dtype=tf.uint8)

    if return_cat_class:
        cat_class = tf.strings.split(path, sep='/')[-3]
        return cat_class, image

    else:
        return image


def resize_image(image, height=256, width=256):
    """Resize tensor jpeg image to height and width"""
    return tf.cast(tf.image.resize_with_pad(image, height, width), tf.uint8)


def image_generators(images_dir, image_size=(256, 256), type='raw', validation_split=0.25, sort_by=None):
    """Return dataset train and validation image generators. Each image will be resized to image_size.
        Args:
            image_size: target size for images. aspect ration will be reserved
            type: which type of images to extract. Available types are 'raw' or 'face', defaults to 'raw'
            sort_by: function by which to sort the cats
            images_dir: images directory
            validation_split: which portion of the dataeset to save for validation

        Returns:
            train_dataset: TF dataset with train images resized to image_size
            validation_dataset: TF dataset with validation images resized to image_size
            dir_object: dict with {train_dirs: list of train dirs, val_dirs: list of validation dirs
                                    , train_size: number of train images, val_size: number of validation images}
    """

    # split train and val dirs
    cat_dirs = get_cat_dirs(images_dir, sort_by=sort_by)
    val_dirs = list(np.random.choice(cat_dirs, size=int(len(cat_dirs) * validation_split)))
    train_dirs = list(set(cat_dirs) - set(val_dirs))

    # get cat images paths train and validation dataset
    train_image_paths = get_cat_image_paths(type=type, cat_paths=train_dirs, sort_by=sort_by)
    val_image_paths = get_cat_image_paths(type=type, cat_paths=val_dirs, sort_by=sort_by)
    train_image_paths = remove_non_jpegs(train_image_paths)  # remove non jpegs from the dataset
    val_image_paths = remove_non_jpegs(val_image_paths)  # remove non jpegs from the dataset

    train_image_paths_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
    val_image_paths_dataset = tf.data.Dataset.from_tensor_slices(val_image_paths)

    # read and resize image to image_size
    train_dataset = train_image_paths_dataset.map(read_image)
    validation_dataset = val_image_paths_dataset.map(read_image)

    # helper function
    def resize_func(cat_class, image):
        return cat_class, resize_image(image, height=image_size[0], width=image_size[1])

    train_dataset = train_dataset.map(resize_func)
    validation_dataset = validation_dataset.map(resize_func)

    # dir object
    dir_obj = {'train_dirs': train_dirs, 'val_dirs': val_dirs, 'train_size': len(train_dirs), 'val_size': len(val_dirs)}

    return train_dataset, validation_dataset, dir_obj


def remove_non_jpegs(filepath_list):
    """Remove non jpeg files from a list of files"""
    for file in filepath_list:
        if imghdr.what(file) != 'jpeg':
            filepath_list.remove(file)

    return filepath_list


if __name__ == '__main__':
    images_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
