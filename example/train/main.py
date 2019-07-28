import argparse
import math
import os
import tensorflow as tf
import example.train.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--config_file')

if __name__ == '__main__':
    args = parser.parse_args()
    args = utils.open_config_file(args.config_file)

    model = utils.get_model(args["model"])
    model = model(shape=args["shape"])
    
    model.compile(loss = utils.get_loss(args["loss"]),
        optimizer = tf.keras.optimizers.Adam(lr=args["learning_rate"]),
        metrics = ['accuracy']
    )

    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir='./graph')
        # TrainValTensorBoard(update_freq='batch')
    ]
    
    train_images, train_labels = utils.load_image_path_and_label(
        os.path.join(os.path.dirname(args["train_list"]), 'train'),
        args["train_list"]
    )
    # test_images, test_labels = load_image_path_and_label(
    #     os.path.join(os.path.dirname(args["test_list"]), 'test'),
    #     args["test_list"]
    # )
    shape = [args["batch_size"]] + args["shape"]
    num_iter = int(math.floor(len(train_images) / args["batch_size"]))
    ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds = ds.map(utils.load_and_preprocess_image)
    # ds = ds.shuffle(buffer_size=4096)
    ds = ds.repeat()
    ds = ds.batch(int(args["batch_size"]))
    ds = ds.prefetch(buffer_size=5000)
    model.fit(
        ds,
        epochs = args["epoch"],
        steps_per_epoch = num_iter
    )