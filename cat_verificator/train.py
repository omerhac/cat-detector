from modules import *
from triplet_loss import _get_anchor_negative_triplet_mask, batch_all_triplet_loss, batch_hard_triplet_loss
import etl
import os
import glob

# constants
MARGIN = 0.1


@tf.function
def loss_function(labels, embeddings, alpha):
    """
    Loss function as described in http://cs230.stanford.edu/projects_fall_2019/reports/26251543.pdf.
    Essentially triplet loss with global orthogonal regulrization.
    """

    # get dot products
    dot_product = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    dot_product_squared = tf.math.pow(dot_product, 2)

    # get negative pairs mask
    neg_mask = _get_anchor_negative_triplet_mask(labels)
    neg_mask = tf.cast(neg_mask, tf.float32)
    num_pairs = tf.reduce_sum(neg_mask)

    # get regularization terms
    m1 = tf.reduce_sum(dot_product * neg_mask) / num_pairs
    m2 = tf.reduce_sum(dot_product_squared * neg_mask) / num_pairs

    # get embeddings dimension
    dim_term = tf.cast(1 / tf.shape(embeddings)[1], tf.float32)

    # compute global orthogonal regularization term
    l_gor = m1 ** 2 + tf.maximum(tf.constant(0, dtype=tf.float32), m2 - dim_term)

    # get triplet_loss
    l_triplet = batch_hard_triplet_loss(labels, embeddings, MARGIN)

    return l_triplet + alpha * l_gor


def train_stage(model, dataset, optimizer, dataset_size, batch_size, epochs=30):
    """One training stage in the training process.
    Args:
        model: model to train
        dataset: cat images dataset. batched and repeated.
        optimizer: initialized training optimizer
        dataset_size: number of images in the dataset
        batch_size: images batch size
        epochs: number of epochs
    """

    # initialize aggregators
    mean_loss = tf.keras.metrics.Mean(name='mean_loss')

    for epoch in range(epochs):
        print(f'Epoch number {epoch}')

        for batch_num, (batch_labels, batch_images) in enumerate(dataset):
            # calculate and apply gradients
            with tf.GradientTape() as tape:
                batch_embeddings = model(batch_images)
                loss = loss_function(batch_labels, batch_embeddings, MARGIN)  # compute loss

                # apply gradients
                grads_and_vars = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads_and_vars, model.trainable_variables))

            # aggregate
            mean_loss(loss)
            # print loss
            if batch_num % 10 == 0:
                print(f'Batch {batch_num}/{dataset_size//batch_size - 1}, Mean loss is {mean_loss.result()}')

            # finished dataset break rule
            if batch_num == dataset_size // batch_size - 1:
                break


def train():
    """Train cat verificator"""
    # initiate a model
    model = CatVerificator(input_shape=(256, 256, 3))

    # get dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
    dataset, dataset_size = etl.image_generator(base_dir, image_size=(64, 64))
    dataset = dataset.repeat()
    dataset = dataset.batch(385)
    optimizer = tf.keras.optimizers.Adam()

    train_stage(model, dataset, optimizer, dataset_size, 10)


if __name__ == '__main__':
    train()
