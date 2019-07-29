import argparse
import math
import os
import tensorflow as tf
import example.train.utils as utils
import example.train.input_pipeline as input_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--config_file')

if __name__ == '__main__':
    args = parser.parse_args()
    args = utils.open_config_file(args.config_file)

    model_fn = utils.get_model(args["model"])
    model = model_fn(shape=args["shape"])
    loss_fn = utils.get_loss(args["loss"])

    train_ds = input_pipeline.Dataset(
        os.path.join(os.path.dirname(args["train_list"]), 'train'),
        args["train_list"],
        args['train_crop_list']
    ).resize(args["shape"][0:2]).shuffle(1000).batch(args["batch_size"]).prefetch(5000)

    test_ds = input_pipeline.Dataset(
        os.path.join(os.path.dirname(args["test_list"]), 'test'),
        args["test_list"],
        args['test_crop_list']
    ).resize(args["shape"][0:2]).shuffle(1000).batch(args["batch_size"]).prefetch(5000)

    writer = tf.summary.create_file_writer('logs')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Iterate over epochs.
    for epoch in range(args["epoch"]):
        print('Start of epoch %d' % (epoch,))
        # Iterate over the batches of the dataset.
        for step, (x, y) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                features = model(x)
                total_loss = loss_fn(features, y)
            grads = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with writer.as_default():
                tf.summary.scalar('total_loss', total_loss, step)
            if step % 100 == 0:
                print('step %s: mean loss = %s' % (step, total_loss.numpy()))