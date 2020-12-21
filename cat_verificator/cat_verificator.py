from modules import *
import tensorflow as tf
from etl import resize_image, read_image
import os
import matplotlib.pyplot as plt
import utilities
import detect_faces
import pickle
import tensorflow.python.keras.backend as K

# define dir path
dir_path = os.path.dirname(os.path.abspath(__file__))

# disable eager execution
tf.compat.v1.disable_eager_execution()


class CatVerificator():
    """Cat verificator object
    Attributes:
        cat_embedder: CatEmbedder model to embed cat images
        threshold: threshold for separating different and same cats
        data_dir: directory to store application data. structure:
            - images
                - cropped: cropped images
                - raw: raw images
            - own_embedding.dat: pickled own embedding
            - threshold.txt: separating threshold

    Methods:
         set_threhold: set separating threshold
         is_same_cat: check whether 2 images are of the same cat
         is_own_cat: check whether a cat is own cat
         set_own_image: sets the own cat image of the application
         resize_input: resizes input images for CatEmbedder digestion
    """

    def __init__(self, embedder_input_shape, threshold=1.25, data_dir='data', load_data=False):
        """Initialize a cat verficator.
        Args:
            embedder_input_shape: input image shape of cat embedder
            threshold: separating threshold
            data_dir: directory for storing application data
            load_data: whether to load application data from directory
        """
        # load model from last checkpoint
        self._cat_embedder = CatEmbedder(input_shape=embedder_input_shape)
        self._cat_embedder.load_checkpoint(tf.train.latest_checkpoint(dir_path + '/weights/checkpoints'))

        # set attrs
        self._threshold = threshold
        self._data_dir = data_dir
        self._own_embedding = tf.zeros(shape=[1, 64], dtype=tf.float32)
        self._image_to_verify = tf.compat.v1.placeholder(tf.float32, shape=[None, None, 3])
        self._verification_graph = self.create_verification_graph()
        self._sess = K.get_session()

        # load data if required
        if load_data:
            self._threshold = float(open(data_dir + '/threshold.txt', 'r').readline())
            self._own_embedding = tf.constant(pickle.load(open(data_dir + '/own_embedding.dat', 'rb')),
                                              dtype=tf.float32)

        # save data
        else:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            open(data_dir + '/threshold.txt', 'w').write(str(self._threshold))

    def close_session(self):
        """Close current verification session"""
        self._sess.close()

    def set_threshold(self, threshold):
        """Set new threshold to threshold"""
        self._threshold = threshold
        open(self._data_dir + '/threshold.txt', 'w').write(str(self._threshold))

    def resize_input(self, image):
        """Resize image so it fits cat embedder input shape"""
        image_shape = self._cat_embedder.get_input_shape()
        image_height, image_width = image_shape[0], image_shape[1]
        return resize_image(image, height=image_height, width=image_width)

    def create_verification_graph(self):
        """Create the graph used for verification"""
        resized_cat = self.resize_input(self._image_to_verify)
        cat_embedd = self._cat_embedder([resized_cat])

        # get distance
        distance = tf.reduce_sum(tf.pow(cat_embedd - self._own_embedding, 2))
        return distance < self._threshold, cat_embedd

    def is_own_cat(self, cat):
        """Check whether cat and own cat are the same cat. Resize image if necessary.
        Args:
            cat: cropped image of a face of a cat. array/tensor
        """

        # run verification graph
        return self._sess.run(self._verification_graph, feed_dict={self._image_to_verify: cat})

    def set_own_image(self, image):
        """Set own cat image to be image. Also embedds the image and save it as _own_embedding.
        Args:
            image: image as an array
        """

        # create directory for storing images
        raw_images_path = self._data_dir + '/images/raw'
        cropped_images_path = self._data_dir + '/images/cropped'
        if not os.path.exists(raw_images_path):
            os.makedirs(raw_images_path)
        if not os.path.exists(cropped_images_path):
            os.makedirs(cropped_images_path)

        # save own image
        plt.imsave(raw_images_path + '/own.jpg', image)

        # crop and save face
        detect_faces.detect_faces(raw_images_path, cropped_images_path, save_images=False)  # save only the coordinates
        utilities.crop_directory_bounding_boxes(raw_images_path, cropped_images_path,
                                                cropped_images_path + '/Detection_Results.csv')

        # get own image embedding
        cropped_own = plt.imread(cropped_images_path + '/own.jpg')
        input_image = tf.compat.v1.placeholder(tf.float32, shape=[None, None, 3])
        cropped_input = self.resize_input(input_image)  # resize to cat embedder input shape
        embedd = self._cat_embedder(cropped_input)
        self._own_embedding = self._sess.run(embedd, feed_dict={input_image: cropped_own})

        # dump embedding
        pickle.dump(self._own_embedding, open(self._data_dir + '/own_embedding.dat', 'wb'))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
    path1 = base_dir + '/49403512/cropped/1.jpg'
    path2 = base_dir + '/49726525/cropped/1.jpg'
    cat1 = plt.imread(path1)
    cat2 = plt.imread(path2)

    cat_ver = CatVerificator([64, 64, 3], 1.25, 'data', load_data=True)
    #cat_ver.set_own_image(cat1)
    print(cat_ver.is_own_cat(cat2))
    print(pickle.load(open('data/own_embedding.dat', 'rb')))

