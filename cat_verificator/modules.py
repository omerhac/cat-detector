from abc import ABC
import tensorflow as tf
from triplet_loss import _get_anchor_negative_triplet_mask, _get_anchor_positive_triplet_mask, _pairwise_distances
import numpy as np
import os
import etl
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# define dir path
dir_path = os.path.dirname(os.path.abspath(__file__))

# use mixed precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


class CatEmbedder(tf.keras.Model, ABC):
    """Cat embedder model"""

    def __init__(self, input_shape=(256, 256, 3)):
        super(CatEmbedder, self).__init__()

        self._input_shape = input_shape

        # initialize efficienetnet with imagenet weights
        ef_net = tf.keras.applications.EfficientNetB2(include_top=False, pooling='avg', input_shape=input_shape,
                                                      weights=dir_path + '/weights/efficientnetb2_notop.h5')
        ef_net.trainable = False

        self._model = tf.keras.models.Sequential([
            ef_net,
            tf.keras.layers.Dense(64, activation='linear', name='dense_rep'),
            tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1), name='l2_norm')
        ])

    def call(self, x):
        return self._model(x)

    def unfreeze_block(self, block_num):
        """Unfreeze layers from block_num parameters"""
        for layer in self._model.layers[0].layers:
            if layer.name[5] == str(block_num):  # check layers block
                layer.trainable = True  # make layer trainable

    def unfreeze_top(self):
        """Unfreeze top layers after blocks parameters"""
        for layer in self._model.layers[0].layers:
            if layer.name in ['top_conv', 'top_bn', 'top_activation', 'avg_pool']:  # check layer is in top layers
                layer.trainable = True  # make layer trainable

    def unfreeze_all(self):
        """Unfreeze all model weights"""
        self._model.layers[0].trainable = True

    def get_input_shape(self):
        return self._input_shape

    def load_checkpoint(self, ckpt_path=None):
        """Load model weights from ckpt_path, if None restores from latest checkpoint"""
        dummy_opt = tf.keras.optimizers.Adam()
        ckpt = tf.train.Checkpoint(model=self, optimizer=dummy_opt)

        # check where to restore from
        if ckpt_path:
            ckpt.restore(ckpt_path)
            print(f'Restored weights from {ckpt_path}')
        else:
            ckpt_path = tf.train.latest_checkpoint('weights/checkpoints')
            ckpt.restore(ckpt_path)
            print(f'Restored weights from {ckpt_path}')

    def load_model(self, model_path):
        """Load model and weights from model_path"""
        self._model = tf.keras.models.load_model(model_path)

    def save_model(self, save_path):
        """Save model to h5 file"""
        self._model.save(save_path, save_format='h5', include_optimizer=False)


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
    tp_sum = tf.reduce_sum(tf.cast(true_positives, tf.float16))
    tn_sum = tf.reduce_sum(tf.cast(true_negatives, tf.float16))

    # get ground truth
    positives = tf.reduce_sum(tf.cast(positive_mask, tf.float16))
    negatives = tf.reduce_sum(tf.cast(negative_mask, tf.float16))

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


def examine_thresholds(input_shape, cat_embedder=None, type='raw', examples_number=32):
    """Print TPR and FPR for threshold range"""
    if not cat_embedder:
        cat_embedder = CatEmbedder(input_shape=input_shape)
        cat_embedder.load_checkpoint(tf.train.latest_checkpoint('weights/checkpoints'))

    # get dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
    train_dataset, val_dataset, _ = etl.image_generators(base_dir, (input_shape[0], input_shape[1]), type=type)
    train_dataset = train_dataset.batch(examples_number)

    for batch_labels, batch_images in train_dataset.take(1):
        batch_embeddings = cat_embedder(batch_images)
        _, metrics = auc_score(batch_embeddings, batch_labels, return_metrics=True)

    for threshold in metrics:
        print(f'Threshold {threshold}: TPR-{metrics[threshold][0]} FPR-{metrics[threshold][1]}')


if __name__ == '__main__':
    a = CatEmbedder(input_shape=[64, 64, 3])
    for layer in a._model.layers[0].layers:
        print(layer.name, layer.trainable)
    a.unfreeze_top()
    a.unfreeze_block(7)
    for layer in a._model.layers[0].layers:
        print(layer.name, layer.trainable)
    a.unfreeze_block(6)
    for layer in a._model.layers[0].layers:
        print(layer.name, layer.trainable)



