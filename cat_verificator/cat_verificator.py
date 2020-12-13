from modules import *
import tensorflow as tf
from etl import resize_image, read_image
import os
from train import load_checkpoint


class CatVerificator():
    """Cat verificator object
    Attributes:
        cat_embedder: CatEmbedder model to embed cat images
        threshold: threshold for separating different and same cats

    Methods:
         set_threhold: set separating threshold
         is_same_cat: check whether to images are of the same cat
    """

    def __init__(self, cat_embedder, threshold):
        """Initialize a cat verficator.
        Args:
            cat_embedder: CatEmbedder model
            threshold: seperating threshold
        """

        self._cat_embedder = cat_embedder
        self._threshold = threshold

    def set_threshold(self, threshold):
        """Set new threshold to threshold"""
        self._threshold = threshold

    def is_same_cat(self, cat_1, cat_2):
        """Check whether cat_1 and cat_2 are images of the same cat. Resize image if necessary"""
        # get cat embedder input size
        image_shape = self._cat_embedder.get_input_shape()
        image_height, image_width = image_shape[0], image_shape[1]

        cat1_embed = self._cat_embedder(resize_image(cat_1, height=image_height, width=image_width))
        cat2_embed = self._cat_embedder(resize_image(cat_2, height=image_height, width=image_width))

        # get distance
        distance = tf.reduce_sum(tf.pow(cat1_embed - cat2_embed, 2))

        return (distance < self._threshold).numpy()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
    path1 = base_dir + '/49789087/raw/2.jpg'
    path2 = base_dir + '/49789087/raw/4.jpg'
    cat1 = read_image(path1, return_cat_class=False)
    cat2 = read_image(path2, return_cat_class=False)

    cat_embedder = CatEmbedder(input_shape=[64, 64, 3])
    cat_embedder.load_checkpoint('weights/checkpoints/ckpt-7')

    cat_ver = CatVerificator(cat_embedder, 1.25)
    print(cat_ver.is_same_cat(cat1, cat2))