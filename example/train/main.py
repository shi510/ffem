import math
import os

import example.train.utils as utils
import example.train.input_pipeline as input_pipeline
import example.train.config
import model.models
import model.logit_fn.margin_logit as margin_logit

import tensorflow as tf
import numpy as np
from tensorboard.plugins import projector


def softmax_xent_loss_wrap(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    return tf.reduce_mean(loss)


keras_model_custom_obj = {
    'softmax_xent_loss_wrap':
        softmax_xent_loss_wrap,
    'original_triplet_loss':
        utils.get_loss('triplet_loss.original_triplet_loss'),
    'adversarial_triplet_loss':
        utils.get_loss('triplet_loss.adversarial_triplet_loss'),
}


def build_dataset(config):
    ds = input_pipeline.make_RFW_tfdataset(
        config['train_file'],
        config['img_root_path'],
        config['num_identity'],
        config['batch_size'],
        config['shape'][:2],
        config['train_classifier'])
    return ds, config['num_identity']


def build_model(config, num_class):
    net = None
    loss_fn = None

    if os.path.exists(config['saved_model']):
        net = tf.keras.models.load_model(
            config['saved_model'],
            custom_objects=keras_model_custom_obj)
        net.load_weights(config['saved_model']+'/variables/variables')
        net.trainable = True
        print('Loading saved weights, Done.')
        if not config['train_classifier']:
            net = tf.keras.Model(net.input, net.get_layer('batch_normalization').output)
            net.trainable = False
            net.get_layer('dense').trainable = True
            net.get_layer('batch_normalization').trainable = True
            # net = tf.keras.Model(net.input, net.get_layer('flatten').output)
            # net.trainable = False
            # embedding = tf.keras.layers.Dense(
            #     config['embedding_dim'], use_bias=False)(net.output)
            # net = tf.keras.Model(net.input, embedding)
        elif net.output.shape[-1] != num_class:
            label = net.input[1]
            embedding = net.get_layer('batch_normalization').output
            embedding = margin_logit.ArcMarginPenaltyLogists(num_class, logist_scale=15)(embedding, label)
            net = tf.keras.Model(net.input, embedding)

    else :
        net = model.models.get_model(config['model'], config['shape'])
        embedding = tf.keras.layers.Dropout(rate=0.5)(net.output)
        embedding = tf.keras.layers.GlobalAveragePooling2D()(embedding)
        embedding = tf.keras.layers.Flatten()(embedding)
        # embedding = tf.keras.layers.Dropout(rate=0.5)(net.output)
        # embedding = tf.keras.layers.Flatten()(embedding)

        if config['train_classifier']:
            if num_class is None:
                print('num_class == None')
                exit(1)
            label = tf.keras.Input([])
            embedding = tf.keras.layers.Dense(config['embedding_dim'],
                kernel_regularizer=tf.keras.regularizers.l2(5e-4))(embedding)
            embedding = tf.keras.layers.BatchNormalization()(embedding)
            embedding = margin_logit.ArcMarginPenaltyLogists(num_class, logist_scale=15)(embedding, label)
            net = tf.keras.Model([net.input, label], embedding)
            # embedding = tf.keras.layers.Dense(
            #     num_class, name='embedding')(embedding)
        else:
            embedding = tf.keras.layers.Dense(
                config['embedding_dim'], use_bias=False)(embedding)
            net = tf.keras.Model(net.input, embedding)

    if config['train_classifier']:
        loss_fn = softmax_xent_loss_wrap
    else:
        loss_fn = utils.get_loss(config['metric_loss'])
    return net, loss_fn


def build_callbacks(config):
    callback_list = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5,
        patience=10, min_lr=1e-4)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=config['checkpoint_dir'],
        save_weights_only=False,
        monitor='loss',
        mode='auto',
        save_best_only=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    if not config['lr_decay']:
        callback_list.append(reduce_lr)
    callback_list.append(checkpoint)
    callback_list.append(early_stop)
    return callback_list


def visualize_embeddings(config, identities=50):
    pathes, labels, boxes = utils.read_dataset_from_json(config['train_file'])
    indexed_data = []
    for n, (path, label) in enumerate(zip(pathes, labels)):
        if label >= identities:
            break
        indexed_data.append({'path': path, 'label': label})
    model = tf.keras.models.load_model(config['saved_model'],
        custom_objects=keras_model_custom_obj)
    if config['train_classifier']:
        last_out = model.get_layer('dense').output
        embedding_dim = last_out.shape[-1]
    else:
        last_out = model.output
        embedding_dim = config['embedding_dim']
    embeddings = tf.math.l2_normalize(last_out, axis=1)
    model = tf.keras.Model(model.input, embeddings)
    embedding_array = np.ndarray((len(indexed_data), embedding_dim))

    for n, img_path in enumerate(indexed_data):
        img_path = os.path.join(config['img_root_path'], img_path['path'])
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, config['shape'][:2])
        img /= 255 # normalize to [0,1] range
        img = tf.expand_dims(img, 0)
        em = model(img)
        embedding_array[n,] = em.numpy()
    # Set up a logs directory, so Tensorboard knows where to look for files
    log_dir='embedding_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for data in indexed_data:
            f.write("{}\n".format(data['label']))

    # Save the weights we want to analyse as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, so
    # we will remove that value.
    embedding = tf.Variable(embedding_array, name='embedding')
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=embedding)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def build_optimizer(config):
    # In tf-v2.3.0, Do not use tf.keras.optimizers.schedules with ReduceLR callback.
    if config['lr_decay']:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            config['lr'],
            decay_steps=config['lr_decay_steps'],
            decay_rate=config['lr_decay_rate'],
            staircase=True)
    else:
        lr = config['lr']

    opt_list = {
        'adam': 
            tf.keras.optimizers.Adam(learning_rate=lr),
        'sgd':
            tf.keras.optimizers.SGD(learning_rate=lr,
                momentum=0.9, nesterov=True)
    }
    if config['optimizer'] not in opt_list:
        print(config['optimizer'], 'is not support.')
        print('please select one of below.')
        print(opt_list.keys())
        exit(1)
    return opt_list[config['optimizer']]


if __name__ == '__main__':
    config = example.train.config.config
    train_ds, num_class = build_dataset(config)
    model, loss_fn = build_model(config, num_class)
    opt = build_optimizer(config)
    model.summary()

    if config['use_keras']:
        model.compile(optimizer=opt, loss=loss_fn)
        model.fit(train_ds, epochs=config['epoch'], verbose=1,
            workers=10, callbacks=build_callbacks(config))
        model.save('{}.h5'.format(config['model_name']),
            include_optimizer=False)
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

    print('Generating TensorBoard Projector for Embedding Vector...')
    visualize_embeddings(config)