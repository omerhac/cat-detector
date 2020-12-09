from abc import ABC
import tensorflow as tf
from triplet_loss import _get_anchor_negative_triplet_mask, _get_anchor_positive_triplet_mask, _pairwise_distances
import numpy as np


class CatVerificator(tf.keras.Model, ABC):
    """Cat verificator model"""

    def __init__(self, input_shape=(256, 256, 3)):
        super(CatVerificator, self).__init__()

        # initialize efficienetnet with imagenet weights
        self._efnet = tf.keras.applications.EfficientNetB2(include_top=False, pooling='avg', input_shape=input_shape,
                                                           weights='weights/efficientnetb2_notop.h5')
        self._efnet.trainable = False

        self._dense_rep = tf.keras.layers.Dense(64, activation='linear', name='dense_rep')

    def __call__(self, x):
        efnet_out = self._efnet(x)
        dense_rep = self._dense_rep(efnet_out)

        # l2 normalize
        dense_rep = tf.keras.backend.l2_normalize(dense_rep, axis=1)

        return dense_rep

    def unfreeze_block(self, block_num):
        """Unfreeze layers from block_num parameters"""
        for layer in self._efnet.layers:
            if layer.name[5] == str(block_num):  # check layers block
                layer.trainable = True  # make layer trainable

    def unfreeze_top(self):
        """Unfreeze top layers after blocks parameters"""
        for layer in self._efnet.layers:
            if layer.name in ['top_conv', 'top_bn', 'top_activation', 'avg_pool']:  # check layer is in top layers
                layer.trainable = True  # make layer trainable


def threshold_accuracy(threshold, batch_embeddings, batch_labels):
    """Return threshold accuracy on separating embeddings on either being positive or negative.
    The score is calculated as the (true positives + true negatives) / number of predictions
    """

    distance_matrix = _pairwise_distances(batch_embeddings)

    # get masks
    positive_mask = _get_anchor_positive_triplet_mask(batch_labels)
    negative_mask = _get_anchor_negative_triplet_mask(batch_labels)

    # check criteria
    true_positives = tf.logical_and(positive_mask, distance_matrix <= threshold)
    true_negatives = tf.logical_and(negative_mask, distance_matrix > threshold)
    tp_sum = tf.reduce_sum(tf.cast(true_positives, tf.float32))
    tn_sum = tf.reduce_sum(tf.cast(true_negatives, tf.float32))

    # total predictions
    pred_sum = tf.cast(tf.shape(batch_embeddings)[0] ** 2 - tf.shape(batch_embeddings)[0], tf.float32)  # remove self distance

    return (tp_sum + tn_sum) / pred_sum


def find_threshold(batch_embeddings, batch_labels):
    """Return threshold from 0.1 to 1 with the best accuracy and the accuracy score for it"""
    accuracy_agg = []
    thresholds = np.linspace(start=0.1, stop=1, num=10)

    for threshold in thresholds:
        accuracy = threshold_accuracy(threshold, batch_embeddings, batch_labels)  # get accuracy for threshold
        accuracy_agg.append(accuracy)

    return thresholds[np.argmax(accuracy_agg)], np.max(accuracy_agg)


def distance_accuracy(batch_embeddings, batch_labels):
    """Compute the accuracy score for this batch. Uses best threshold for this batch"""
    _, accuracy = find_threshold(batch_embeddings, batch_labels)
    return accuracy


if __name__ == '__main__':
    a = tf.constant([
        [1, 0, 0],
        [5, 0, 0],
        [4, 0, 0]
    ], dtype=tf.float32)
    b = tf.constant([
        1, 1, 0
    ], dtype=tf.float32)
    print(distance_accuracy(a, b))
