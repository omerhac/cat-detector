from modules import *
from triplet_loss import _get_anchor_negative_triplet_mask, batch_all_triplet_loss, batch_hard_triplet_loss
import etl
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import glob

# constants
MARGIN = 0.4
ALPHA = 1.1

# use mixed precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


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
    neg_mask = tf.cast(neg_mask, tf.float16)
    num_pairs = tf.reduce_sum(neg_mask)

    # get regularization terms
    m1 = tf.reduce_sum(dot_product * neg_mask) / num_pairs
    m2 = tf.reduce_sum(dot_product_squared * neg_mask) / num_pairs

    # get embeddings dimension
    dim_term = tf.cast(1 / tf.shape(embeddings)[1], tf.float16)

    # compute global orthogonal regularization term
    l_gor = m1 ** 2 + tf.maximum(tf.constant(0, dtype=tf.float16), m2 - dim_term)

    # get triplet_loss
    l_triplet = batch_hard_triplet_loss(labels, embeddings, MARGIN)

    return l_triplet + alpha * l_gor


def train_stage(model, train_dataset, val_dataset, optimizer, dir_obj, batch_size, ckpt_manager, epochs=30):
    """One training stage in the training process.
    Args:
        model: model to train
        train_dataset: training cat images dataset. batched and repeated.
        val_dataset: validation cat images dataset. batched and repeated.
        optimizer: initialized training optimizer
        dir_obj: dictionary containing training and validation directory names and sizes
        batch_size: images batch size
        epochs: number of epochs
        ckpt_manager: checkpoint manager
    """

    # initialize aggregators
    mean_loss = tf.keras.metrics.Mean(name='mean_loss')
    mean_auc = tf.keras.metrics.Mean(name='mean_auc')

    # get dataset size
    train_size = dir_obj['train_size']
    val_size = dir_obj['val_size']

    for epoch in range(epochs):
        print(f'Epoch number {epoch}')

        # train
        for batch_num, (batch_labels, batch_images) in enumerate(train_dataset):
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
                print(f'Batch {batch_num}/{train_size//batch_size - 1}, Mean loss is {mean_loss.result()}, Mean AUC is {mean_auc.result()}')

            # finished dataset break rule
            if batch_num == train_size // batch_size - 1:
                break

        # evaluate
        val_auc = 0
        for batch_num, (batch_labels, batch_images) in enumerate(val_dataset):
            batch_embeddings = model(batch_images)
            # aggregate metric
            val_auc += auc_score(batch_embeddings, batch_labels)
            # break rule
            if batch_num == val_size // batch_size - 1:
                break

        print(f'Validation AUC score is: {val_auc / (val_size // batch_size)}')

        # save model
        if epoch % 5 == 0:
            save_path = ckpt_manager.save()
            print("Saved model after {} epochs to {}".format(epoch, save_path))

        # shuffle, its ugly I know.. but it has to only shuffle directory order
        train_dataset, _, _ = etl.create_datasets_from_directories_list(dir_obj['train_dirs'], [],
                                                                        image_size=[256, 256], type='cropped',
                                                                        shuffle=True)


def train(image_shape=[256, 256], load_dir='weights/checkpoints'):
    """Train cat embedder
    Args:
        image_shape: size which to resize the images to
        load_dir: checkpoints directory
    """

    # initiate a model
    model = CatEmbedder(input_shape=[*image_shape, 3])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # load checkpoint
    manager = load_checkpoint(model, optimizer, load_dir=load_dir)

    # get dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
    train_dataset, val_dataset, dir_obj = etl.image_generators(base_dir, image_size=image_shape, type='cropped',
                                                               validation_split=0.2)

    # first training stage
    batch_size = 128  # 768
    train_dataset = train_dataset.repeat()
    train_dataset_batched = train_dataset.batch(batch_size)
    val_dataset = val_dataset.repeat()
    val_dataset_batched = val_dataset.batch(batch_size)

    train_stage(model, train_dataset_batched, val_dataset_batched, optimizer, dir_obj, batch_size, manager, epochs=10)

    # save weights
    weights_path = 'weights/cat_embedder_stage1.h5'
    model.save_model(weights_path)
    print(f'Saved model weights at: {weights_path}')

    # second training stage
    batch_size = 128  # 612
    train_dataset_batched = train_dataset.batch(batch_size)
    val_dataset_batched = val_dataset.batch(batch_size)

    # unfreeze top and block 7 weights
    model.unfreeze_top()
    model.unfreeze_block(7)

    train_stage(model, train_dataset_batched, val_dataset_batched, optimizer, dir_obj, batch_size, manager, epochs=30)

    # save weights
    weights_path = 'weights/cat_embedder_stage2.h5'
    model.save_model(weights_path)
    print(f'Saved model weights at: {weights_path}')

    # third training stage
    batch_size = 128
    train_dataset_batched = train_dataset.batch(batch_size)
    val_dataset_batched = val_dataset.batch(batch_size)

    # unfreeze block 6 weights
    model.unfreeze_block(6)

    train_stage(model, train_dataset_batched, val_dataset_batched, optimizer, dir_obj, batch_size, manager, epochs=30)

    # save weights
    weights_path = 'weights/cat_embedder_stage3.h5'
    model.save_model(weights_path)
    print(f'Saved model weights at: {weights_path}')

    # fourth training stage
    batch_size = 48
    train_dataset_batched = train_dataset.batch(batch_size)
    val_dataset_batched = val_dataset.batch(batch_size)

    # unfreeze all weights
    model.unfreeze_all()

    #train_stage(model, train_dataset_batched, val_dataset_batched, optimizer, dir_obj, batch_size, manager, epochs=45)

    # save weights
    weights_path = 'weights/cat_embedder_final.h5'
    model.save_model(weights_path)
    print(f'Saved model weights at: {weights_path}')


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
    train(image_shape=[256, 256])