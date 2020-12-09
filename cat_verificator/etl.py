import tensorflow as tf
import os
import imghdr
import glob


def get_cat_dirs(base_dir, sort_by=None):
    """
    Return a list with cat directories.
    Args:
        base_dir: baseline dir for images
        sort_by: function to sort the cats by
    """

    return  sorted(glob.glob(base_dir + '/*'), key=sort_by)


def get_cat_image_paths(cat_paths=None, type='raw', images_dir=None, sort_by=None):
    """
    Return a list with cats images from a list of cats paths.
    Args:
        cat_paths: list with cats directories paths
        type: which type of images to extract. Available types are 'raw' or 'face', defaults to 'raw
        images_dir: base images dir, required only if cat_paths is not provided
        sort_by: function by which to sort the cats
    """

    # cat cat dirs if not provided
    if cat_paths is None:
        images_dir = images_dir
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
        print(path)

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
            images_dir: images directory
    """

    # get cat images paths dataset
    image_paths = get_cat_image_paths(type=type, images_dir=images_dir, sort_by=sort_by)
    image_paths = remove_non_jpegs(image_paths)  # remove non jpegs from the dataset
    image_paths_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # read and resize image to image_size
    image_dataset = image_paths_dataset.map(read_image)

    # helper function
    def resize_func(cat_class, image):
        return cat_class, resize_image(image, height=image_size[0], width=image_size[1])

    image_dataset = image_dataset.map(resize_func)

    return image_dataset, len(image_paths)  # return also dataset size


def remove_non_jpegs(filepath_list):
    """Remove non jpeg files from a list of files"""
    for file in filepath_list:
        if imghdr.what(file) != 'jpeg':
            filepath_list.remove(file)

    return filepath_list


if __name__ == '__main__':
    images_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
    images = glob.glob(images_dir + '/*/raw/*.jpg')
    print(len(images))
    images = remove_non_jpegs(images)
    print(len(images))
    print('finished')
