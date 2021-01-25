import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import train.input_pipeline as input_pipeline
import train.config
from train.callbacks import LogCallback
from train.callbacks import RecallCallback
import net_arch.models
import train.blocks
from train.utils import pairwise_distance
from train.custom_model import ArcFaceModel, CenterSoftmaxModel, ProxyModel

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.mixed_precision as mixed_precision


def build_dataset(config):
    train_ds, test_ds_dict = input_pipeline.make_tfdataset(
        config['train_file'],
        config['test_files'],
        config['batch_size'],
        config['shape'][:2])
    return train_ds, test_ds_dict


def build_backbone_model(config):

    if os.path.exists(config['saved_model']):
        net = tf.keras.models.load_model(config['saved_model'])
        if os.path.isdir(config['saved_model']):
            net.load_weights(config['saved_model']+'/variables/variables')
        print('')
        print('******************** Loaded saved weights ********************')
        print('')
    elif len(config['saved_model']) != 0:
        print(config['saved_model'] + ' can not open.')
        exit(1)
    else :
        net = net_arch.models.get_model(config['model'], config['shape'])

    return net


def build_model(config):
    y = x1 = tf.keras.Input(config['shape'])
    y = build_backbone_model(config)(y)

    def _embedding_layer(feature):
        y = x = tf.keras.Input(feature.shape[1:])
        y = tf.keras.layers.Dropout(rate=0.5)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Flatten()(y)
        y = train.blocks.attach_embedding_projection(y, config['embedding_dim'])
        return tf.keras.Model(x, y, name='embeddings')(feature)


    y = _embedding_layer(y)
    loss_param = copy.deepcopy(config['loss_param'][config['loss']])
    loss_param['n_embeddings'] = config['embedding_dim']
    loss_param['n_classes'] = config['num_identity']
    if config['loss'] == 'CenterSoftmax':
        return CenterSoftmaxModel(inputs=x1, outputs=y, **loss_param)
    elif config['loss'] == 'ProxySoftmax':
        return ProxyModel(inputs=x1, outputs=y, **loss_param)
    elif config['loss'] == 'AddictiveMargin':
        return ArcFaceModel(inputs=x1, outputs=y, **loss_param)


def build_callbacks(config, test_ds_dict):
    log_dir = os.path.join('logs', config['model_name'])
    callback_list = []
    metric_fn = pairwise_distance
    if config['loss'] == 'AddictiveMargin':
        metric_fn = lambda A, B: tf.matmul(A, tf.transpose(B))
    recall_eval = RecallCallback(test_ds_dict, [1], metric_fn, log_dir)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='recall@1', factor=0.1, mode='max',
        patience=1, min_lr=1e-4)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint'+os.path.sep+config['model_name'],
        save_weights_only=False,
        monitor='recall@1',
        mode='max',
        save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='recall@1',
        mode='max', patience=3,
        restore_best_weights=True)
    tensorboard_log = LogCallback(log_dir)

    callback_list.append(recall_eval)
    callback_list.append(tensorboard_log)
    callback_list.append(checkpoint)
    callback_list.append(early_stop)
    if not config['lr_decay']:
        callback_list.append(reduce_lr)
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
        'Adam': 
            tf.keras.optimizers.Adam(learning_rate=lr),
        'SGD':
            tf.keras.optimizers.SGD(learning_rate=lr,
                momentum=0.9, nesterov=True),
        'AdamW': 
            tfa.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
    }
    if config['optimizer'] not in opt_list:
        print(config['optimizer'], 'is not support.')
        print('please select one of below.')
        print(opt_list.keys())
        exit(1)
    return opt_list[config['optimizer']]


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


def remove_subclassing_keras_model(model):
    y = x = model.layers[0].input
    for l in model.layers[1:]:
        y = l(y)
    return tf.keras.Model(x, y, name=model.name)


if __name__ == '__main__':
    config = train.config.config
    if config['mixed_precision']:
        print('---------------- Enabled Mixed Precision ----------------')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    train_ds, test_ds_dict = build_dataset(config)
    net = build_model(config)
    opt = build_optimizer(config)
    net.summary()
    net.compile(optimizer=opt)
    net.fit(train_ds, epochs=config['epoch'], verbose=1,
        workers=input_pipeline.TF_AUTOTUNE,
        callbacks=build_callbacks(config, test_ds_dict))
    net = remove_subclassing_keras_model(net)
    net.save('{}.h5'.format(config['model_name']), include_optimizer=False)

    # print('Generating TensorBoard Projector for Embedding Vector...')
    # utils.visualize_embeddings(config)
