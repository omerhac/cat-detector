from modules import *
from triplet_loss import _get_anchor_negative_triplet_mask, batch_all_triplet_loss, batch_hard_triplet_loss
import etl
import os

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


def train_stage(model, dataset, optimizer, epcohs=30):
    """One training stage in the training process"""
    for epoch in range(epcohs):
        print(f'Epoch number {epoch}')

        for batch_num, (batch_labels, batch_images) in enumerate(dataset):
            # calculate and apply gradients
            with tf.GradientTape() as tape:
                batch_embeddings = model(batch_images)
                loss, _ = batch_all_triplet_loss(batch_labels, batch_embeddings, MARGIN)

                # apply
                grads_and_vars = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads_and_vars, model.trainable_variables))
            # print loss
            if batch_num % 1 == 0:  # TODO: change to 50
                print(f'Current loss is {loss.numpy()}')


def train():
    """Train cat verificator"""
    # initiate
    model = CatVerificator(input_shape=(64, 64, 3))
    images_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = etl.image_generator(images_dir, image_size=(64, 64)).batch(10)
    optimizer = tf.keras.optimizers.Adam()

    train_stage(model, dataset, optimizer)


if __name__ == '__main__':
    a = tf.constant([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1]
    ], dtype=tf.float32)
    b = tf.constant([
        0, 1, 1
    ])
    print(loss_function(b, a, 0.5))
