from modules import *
from triplet_loss import *
import etl
import os

# constants
MARGIN = 0.1


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
            if batch_num % 50 == 0:
                print(f'Current loss is {loss.numpy()}')


def train():
    """Train cat verificator"""
    # initiate
    model = CatVerificator()
    images_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = etl.image_generator(images_dir).batch(10)
    optimizer = tf.keras.optimizers.Adam()

    train_stage(model, dataset, optimizer)


if __name__ == '__main__':
    train()
