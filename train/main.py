import os

import train.input_pipeline as input_pipeline
import train.config
from train.callbacks import LossTensorBoard
import net_arch.models
import train.blocks

import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
import numpy as np


def build_dataset(config):
    ds, num_id = input_pipeline.make_tfdataset(
        config['tfrecord_file'],
        config['num_identity'],
        config['batch_size'],
        config['shape'][:2],
        config['arc_margin_penalty'])
    return ds, num_id


def build_backbone_model(config):
    """
        The saved model should contain the layers below:
            1. Input Layer
            2. Backbone Net layer
            3. Embeeding Layer
    """

    if os.path.exists(config['saved_model']):
        net = tf.keras.models.load_model(config['saved_model'])
        if os.path.isdir(config['saved_model']):
            net.load_weights(config['saved_model']+'/variables/variables')
        if len(net.layers) is not 2:
            y = x = net.layers[0].input
            y = net.layers[1](y)
            net = tf.keras.Model(x, y, name = net.name)
        print('')
        print('******************** Loaded saved weights ********************')
        print('')
    elif len(config['saved_model']) != 0:
        print(config['saved_model'] + ' can not open.')
        exit(1)
    else :
        net = net_arch.models.get_model(config['model'], config['shape'])

    return net


def build_model(config, num_id):
    y = x1 = tf.keras.Input(config['shape'])
    y = build_backbone_model(config)(y)

    def _embedding_layer(feature):
        y = x = tf.keras.Input(feature.shape[1:])
        y = tf.keras.layers.Dropout(rate=0.5)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Flatten()(y)
        y = train.blocks.attach_embedding_projection(y, config['embedding_dim'])
        return tf.keras.Model(x, y, name='embeddings')(feature)


    def _classification_layer(feature):
        y = x = tf.keras.Input(feature.shape[1:])
        y = train.blocks.attach_l2_norm_features(y, scale=30)
        y = tf.keras.layers.Dense(num_id)(y)
        return tf.keras.Model(x, y, name='softmax_classifier')(feature)


    def _arccos_margin_layer(feature, label):
        y = x = tf.keras.Input(feature.shape[1:])
        y = train.blocks.attach_arc_margin_penalty(
            y, label, num_id, scale=60)
        return tf.keras.Model([x, label], y, name='arc_margin')((feature, label))


    y = _embedding_layer(y)
    net_inputs = []
    net_outputs = []
    if config['arc_margin_penalty']:
        x2 = tf.keras.Input([])
    y = _arccos_margin_layer(y, x2)
        net_inputs = [x1, x2]
        net_outputs = y
    else:
        y = _classification_layer(y)
        net_inputs = x1
        net_outputs = y
    return tf.keras.Model(net_inputs, net_outputs, name=config['model_name'])


def build_callbacks(config):
    callback_list = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1,
        patience=1, min_lr=1e-4)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint'+os.path.sep+config['model_name'],
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    tensorboard_log = LossTensorBoard(
        100, os.path.join('logs', config['model_name']))

    if not config['lr_decay']:
        callback_list.append(reduce_lr)
    callback_list.append(checkpoint)
    callback_list.append(early_stop)
    callback_list.append(tensorboard_log)
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


def softmax_xent_loss(y_true, y_pred):
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.reshape(y_true, (-1,))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    return tf.reduce_mean(loss)


def save_model(name, net):
    """
        net.layers[0] is a keras.Input layer.
        net.layers[1] is a Backbone layer.
        net.layers[2] is an Embedding layer.
    """
    y = x = net.layers[0].input
    y = net.layers[1](y)
    y = net.layers[2](y)
    backbone = tf.keras.Model(x, y, name = net.name)
    backbone.save('{}.h5'.format(name), include_optimizer=False)


def convert_tflite_int8(model, ds):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset_gen():
        for n, (x, _ )in enumerate(ds.take(10000)):
            if n % 100 == 0:
                print(n)
            # Get sample input data as a numpy array in a method of your choosing.
            # The batch size should be 1.
            yield [x[0]]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8  # or tf.int8
    tflite_quant_model = converter.convert()
    with open(model.name + '.tflite', 'wb') as f:
        f.write(tflite_quant_model)


if __name__ == '__main__':
    config = train.config.config
    if config['mixed_precision']:
        print('---------------- Enabled Mixed Precision ----------------')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    train_ds, num_id = build_dataset(config)
    net = build_model(config, num_id)
    opt = build_optimizer(config)
    net.summary()

    if config['use_keras']:
        net.compile(optimizer=opt, loss=softmax_xent_loss)
        net.fit(train_ds, epochs=config['epoch'], verbose=1,
            workers=input_pipeline.TF_AUTOTUNE,
            callbacks=build_callbacks(config))
        save_model(config['model_name'], net)
    else:
        # Iterate over epochs.
        for epoch in range(config['epoch']):
            print('Start of epoch %d' % epoch)
            # Iterate over the batches of the dataset.
            epoch_loss = np.zeros(1, dtype=np.float32)
            for step, (x, y) in enumerate(train_ds):
                with tf.GradientTape() as tape:
                    features = net(x)
                    total_loss = softmax_xent_loss(y, features)
                epoch_loss += total_loss.numpy()
                grads = tape.gradient(total_loss, net.trainable_weights)
                opt.apply_gradients(zip(grads, net.trainable_weights))
                # with writer.as_default():
                #     tf.summary.scalar('total_loss', total_loss, step)
                if (step+1) % 100 == 0:
                    print('step %s: mean loss = %s' % (step, epoch_loss / step))
        save_model(config['model_name'], net, config['arc_margin_penalty'])

    # print('Generating TensorBoard Projector for Embedding Vector...')
    # utils.visualize_embeddings(config)