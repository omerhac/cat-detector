import tensorflow as tf
from triplet_loss import *

if __name__ == '__main__':
    a = tf.constant([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ], dtype=tf.float32)
    print(batch_all_triplet_loss([0, 1, 0], a, 0.2))