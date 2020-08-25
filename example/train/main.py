import math
import os
import tensorflow as tf
import numpy as np
import example.train.utils as utils
import example.train.input_pipeline as input_pipeline
import example.train.config
import model.models


def build_dataset(config):
    if config['dataset'] == "RFW":
        rfw_config = config['__{}'.format("RFW")]
        ds, num_class = input_pipeline.make_RFW_tfdataset(
            root_path=rfw_config['root_path'],
            race_list=rfw_config['race_list'],
            num_id=rfw_config['num_identity'],
            batch_size=config['batch_size'],
            img_shape=config['shape'][:2],
            onehot=config['train_classifier'])
    else:
        print('{} is wrong dataset.'.format(config['dataset']))
        exit(1)
    return ds, num_class

def build_model(config, num_class):
    model = None
    loss_fn = None
    if config['train_classifier']:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits
    else:
        loss_fn = utils.get_loss(config['metric_loss'])

    if os.path.exists('{}.h5'.format(config['model_name'])):
        model = tf.keras.models.load_model(
            '{}.h5'.format(config['model_name']),
            custom_objects={
                'softmax_cross_entropy_with_logits_v2':
                tf.nn.softmax_cross_entropy_with_logits})
    else :
        model = model.models.get_model(config['model'], config['shape'])
        embedding = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        embedding = tf.keras.layers.Flatten()(embedding)

        if config['train_classifier']:
            if num_class is None:
                print('num_class == None')
                exit(1)
            embedding = tf.keras.layers.Dense(
                num_class, use_bias=False, name='embedding')(embedding)
        else:
            embedding = tf.keras.layers.Dense(
                config['embedding_dim'], use_bias=False)(embedding)

        model = tf.keras.Model(model.input, embedding)
    return model, loss_fn


if __name__ == '__main__':
    config = example.train.config.config
    train_ds, num_class = build_dataset(config)
    model, loss_fn = build_model(config, num_class)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    if config['use_keras'] and True:
        model.compile(optimizer=opt, loss=loss_fn)
        model.fit(train_ds, epochs=config['epoch'], verbose=1,
            workers=10)
        model.save('{}.h5'.format(config['model_name']))
    else:
        # Iterate over epochs.
        for epoch in range(config['epoch']):
            print('Start of epoch %d' % epoch)
            # Iterate over the batches of the dataset.
            epoch_loss = np.zeros(1, dtype=np.float32)
            for step, (x, y) in enumerate(train_ds):
                with tf.GradientTape() as tape:
                    features = model(x)
                    total_loss = loss_fn(y, features)
                epoch_loss += total_loss.numpy()
                grads = tape.gradient(total_loss, model.trainable_weights)
                opt.apply_gradients(zip(grads, model.trainable_weights))
                # with writer.as_default():
                #     tf.summary.scalar('total_loss', total_loss, step)
                if (step+1) % 100 == 0:
                    print('step %s: mean loss = %s' % (step, epoch_loss / step))
            model.save('{}_{}.h5'.format(config['model_name'], epoch+1))