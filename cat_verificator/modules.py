from abc import ABC
import tensorflow as tf
from triplet_loss import _get_anchor_negative_triplet_mask, _get_anchor_positive_triplet_mask, _pairwise_distances
import numpy as np


class CatEmbedder(tf.keras.Model, ABC):
    """Cat embedder model"""

    def __init__(self, input_shape=(256, 256, 3)):
        super(CatEmbedder, self).__init__()

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


def threshold_metrics(threshold, batch_embeddings, batch_labels):
    """Return threshold TPR and FPR on separating embeddings on either being positive or negative.
    TPR is true positives / predicted positives, FPR is false positives / predicted negatives.
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

    # get ground truth
    positives = tf.reduce_sum(tf.cast(positive_mask, tf.float32))
    negatives = tf.reduce_sum(tf.cast(negative_mask, tf.float32))

    # stats
    eps = 1e-12  # for division
    tpr = tp_sum / (positives + eps)
    tnr = tn_sum / (negatives + eps)
    fpr = 1 - tnr

    return tpr, fpr


def auc_score(batch_embeddings, batch_labels, return_metrics=False):
    """Compute AUC score for embeddings and labels.
    Args:
        batch_embeddings: images embedded by the model
        batch_labels: labels for each image
        return_metrics: if True returns dict of shape {threshold: (TPR@threshold, FPR@threshold)}
    """

    tprs, fprs = [], []
    thresholds = np.linspace(start=0.1, stop=2, num=20)

    # get metrics for different thresholds
    for threshold in thresholds:
        tpr, fpr = threshold_metrics(threshold, batch_embeddings, batch_labels)  # get metrics for threshold
        tprs.append(tpr)
        fprs.append(fpr)

    # compute area
    area = 0
    prev_fpr = 0
    for tpr, fpr in zip(tprs, fprs):
        area += tpr * (fpr - prev_fpr)
        prev_fpr = fpr

    # build metrics dict if needed
    if return_metrics:
        metrics_dict = {}
        for i, threshold in enumerate(thresholds):
            metrics_dict[threshold] = (tprs[i], fprs[i])

        return area, metrics_dict

    else:
        return area


if __name__ == '__main__':
    a = tf.constant([
        [1, 0, 0],
        [2, 0, 0],
        [2.1, 0, 0],
        [4, 0, 0]
    ], dtype=tf.float32)
    b = tf.constant([
        1, 1,1, 0
    ], dtype=tf.float32)
    print(auc_score(a, b))
