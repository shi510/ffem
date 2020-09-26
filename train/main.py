import math
import os

import train.utils as utils
import train.input_pipeline as input_pipeline
import train.config
import model.models

import tensorflow as tf
import numpy as np


def build_dataset(config):
    ds = input_pipeline.make_RFW_tfdataset(
        config['train_file'],
        config['img_root_path'],
        config['num_identity'],
        config['batch_size'],
        config['shape'][:2],
        config['train_classifier'])
    return ds


def build_backbone_model(config):
    if os.path.exists(config['saved_model']):
        net = tf.keras.models.load_model(
            config['saved_model'],
            custom_objects=utils.keras_model_custom_obj)
        if os.path.isdir(config['saved_model']):
            net.load_weights(config['saved_model']+'/variables/variables')
            print('Loading saved weights, Done.')
    else :
        net = model.models.get_model(config['model'], config['shape'])

    return net


def build_embedding_model(config, net):
    net.trainable = True
    classes = config['num_identity']

    # Check if the model is pretrained by a name of the last layer.
    # Because we always attach an auxiliary layer to the top of a backbone model.
    is_pretrained = 'last_flatten' not in net.output.name
    if is_pretrained:
        if not config['train_classifier']:
            net = tf.keras.Model(net.input, net.get_layer('embedding').output)
        elif net.output.shape[-1] != classes:
            net = tf.keras.Model(net.input, net.get_layer('embedding').output)
            if config['arc_margin_penalty']:
                net = model.models.attach_arc_margin_penalty(net, classes, 30)
            else:
                net = model.models.attach_classifier(net, classes)

    else: # not pretrained, so attach auxiliary layer.
        if config['train_classifier']:
            net = model.models.attach_embedding(net, config['embedding_dim'])
            if config['arc_margin_penalty']:
                net = model.models.attach_arc_margin_penalty(net, classes, 30)
            else:
                net = model.models.attach_classifier(net, classes)
        else:
            embedding = tf.keras.layers.Dense(
                config['embedding_dim'], use_bias=False)(embedding)
            net = tf.keras.Model(net.input, embedding)

    return net


def build_callbacks(config):
    callback_list = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5,
        patience=10, min_lr=1e-4)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=config['checkpoint_dir'],
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    if not config['lr_decay']:
        callback_list.append(reduce_lr)
    callback_list.append(checkpoint)
    callback_list.append(early_stop)
    return callback_list


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


def build_loss_fn(config):
    if config['train_classifier']:
        loss_fn = model.loss_fn.default_loss.softmax_xent_loss_wrap
    else:
        loss_fn = utils.get_loss(config['metric_loss'])

    return loss_fn


if __name__ == '__main__':
    config = train.config.config
    net = build_backbone_model(config)
    net = build_embedding_model(config, net)
    loss_fn = build_loss_fn(config)
    opt = build_optimizer(config)
    net.summary()
    train_ds = build_dataset(config)

    if config['use_keras']:
        net.compile(optimizer=opt, loss=loss_fn)
        net.fit(train_ds, epochs=config['epoch'], verbose=1,
            workers=input_pipeline.TF_AUTOTUNE,
            callbacks=build_callbacks(config),
            shuffle=True)
        net.save('{}.h5'.format(config['model_name']))
    else:
        # Iterate over epochs.
        for epoch in range(config['epoch']):
            print('Start of epoch %d' % epoch)
            # Iterate over the batches of the dataset.
            epoch_loss = np.zeros(1, dtype=np.float32)
            for step, (x, y) in enumerate(train_ds):
                with tf.GradientTape() as tape:
                    features = net(x)
                    total_loss = loss_fn(y, features)
                epoch_loss += total_loss.numpy()
                grads = tape.gradient(total_loss, net.trainable_weights)
                opt.apply_gradients(zip(grads, net.trainable_weights))
                # with writer.as_default():
                #     tf.summary.scalar('total_loss', total_loss, step)
                if (step+1) % 100 == 0:
                    print('step %s: mean loss = %s' % (step, epoch_loss / step))
            net.save('{}_{}.h5'.format(config['model_name'], epoch+1))

    print('Generating TensorBoard Projector for Embedding Vector...')
    utils.visualize_embeddings(config)