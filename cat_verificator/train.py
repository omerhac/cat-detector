from modules import *
from triplet_loss import _get_anchor_negative_triplet_mask, batch_all_triplet_loss, batch_hard_triplet_loss
import etl
import os
import glob

# constants
MARGIN = 0.4
ALPHA = 1.1


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


def train_stage(model, dataset, optimizer, dataset_size, batch_size, ckpt_manager, epochs=30):
    """One training stage in the training process.
    Args:
        model: model to train
        dataset: cat images dataset. batched and repeated.
        optimizer: initialized training optimizer
        dataset_size: number of images in the dataset
        batch_size: images batch size
        epochs: number of epochs
        ckpt_manager: checkpoint manager
    """

    # initialize aggregators
    mean_loss = tf.keras.metrics.Mean(name='mean_loss')
    mean_auc = tf.keras.metrics.Mean(name='mean_auc')

    for epoch in range(epochs):
        print(f'Epoch number {epoch}')

        for batch_num, (batch_labels, batch_images) in enumerate(dataset):
            # calculate and apply gradients
            with tf.GradientTape() as tape:
                batch_embeddings = model(batch_images)
                loss = loss_function(batch_labels, batch_embeddings, ALPHA)  # compute loss

                # apply gradients
                grads_and_vars = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads_and_vars, model.trainable_variables))

            # calculate metric
            auc = auc_score(batch_embeddings, batch_labels)

            # aggregate
            mean_loss(loss)
            mean_auc(auc)

            # print loss and accuracy
            if batch_num % 10 == 0:
                print(f'Batch {batch_num}/{dataset_size//batch_size - 1}, Mean loss is {mean_loss.result()}, Mean AUC is {mean_auc.result()}')

            # finished dataset break rule
            if batch_num == dataset_size // batch_size - 1:
                break

        # save model
        if epoch % 5 == 0:
            save_path = ckpt_manager.save()
            print("Saved model after {} epochs to {}".format(epoch, save_path))


def train(image_shape=[256, 256], load_dir='weights/checkpoints'):
    """Train cat verificator
    Args:
        image_shape: size which to resize the images to
        load_dir: checkpoints directory
    """

    # initiate a model
    model = CatEmbedder(input_shape=[*image_shape, 3])

    # get dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
    dataset, dataset_size = etl.image_generator(base_dir, image_size=image_shape)

    batch_size = 64
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # load checkpoint
    manager = load_checkpoint(model, optimizer, load_dir=load_dir)
    train_stage(model, dataset, optimizer, dataset_size, batch_size, manager, epochs=10)


def load_checkpoint(model, optimizer=None, load_dir='checkpoints'):
    """Load model and optimizer from load dir. Return checkpoints manager"""
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, load_dir, max_to_keep=10)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    return manager


if __name__ == '__main__':
    train(image_shape=[64, 64])