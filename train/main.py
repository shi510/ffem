import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import train.input_pipeline as input_pipeline
import train.config
from train.callbacks import LogCallback
from train.callbacks import RecallCallback
from train.utils import apply_pruning
import net_arch.models
import train.blocks
from train.custom_model import AdditiveAngularMarginModel
from train.custom_model import CenterSoftmaxModel
from train.custom_model import ProxyNCAModel

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.mixed_precision as mixed_precision
import tensorflow_model_optimization as tfmot


def build_dataset(config):
    train_ds, test_ds_dict = input_pipeline.make_tfdataset(
        config['train_file'],
        config['test_files'],
        config['batch_size'],
        config['shape'][:2])
    return train_ds, test_ds_dict


def build_backbone_model(config):
    is_pretrained = False
    if os.path.exists(config['saved_model']):
        net = tf.keras.models.load_model(config['saved_model'])
        if os.path.isdir(config['saved_model']):
            net.load_weights(config['saved_model']+'/variables/variables')
        print('')
        print('******************** Loaded saved weights ********************')
        print('')
        is_pretrained = True
    elif len(config['saved_model']) != 0:
        print(config['saved_model'] + ' can not open.')
        exit(1)
    else :
        net = net_arch.models.get_model(config['model'], config['shape'])

    return net, is_pretrained


def build_model(config):
    net, is_pretrained = build_backbone_model(config)
    if config['enable_prune']:
        net = apply_pruning(
            net, config['prune_params'], 1 if is_pretrained else None)

    def _embedding_layer(feature):
        y = x = tf.keras.Input(feature.shape[1:])
        y = train.blocks.attach_GNAP(y)
        y = train.blocks.attach_embedding_projection(y, config['embedding_dim'])
        return tf.keras.Model(x, y, name='embeddings')(feature)

    if not is_pretrained:
        y = x1 = tf.keras.Input(config['shape'])
        y = net(y)
        y = _embedding_layer(y)
    else:
        x1 = net.inputs
        y = net.outputs
    loss_param = copy.deepcopy(config['loss_param'][config['loss']])
    loss_param['n_embeddings'] = config['embedding_dim']
    loss_param['n_classes'] = config['num_identity']
    if config['loss'] == 'CenterSoftmax':
        return CenterSoftmaxModel(inputs=x1, outputs=y, **loss_param)
    elif config['loss'] == 'ProxyNCA':
        return ProxyNCAModel(inputs=x1, outputs=y, **loss_param)
    elif config['loss'] == 'AdditiveAngularMargin':
        return AdditiveAngularMarginModel(inputs=x1, outputs=y, **loss_param)


def build_callbacks(config, test_ds_dict):
    log_dir = os.path.join('logs', config['model_name'])
    callback_list = []
    metric = config['eval']['metric']
    recall_topk = config['eval']['recall']
    recall_eval = RecallCallback(test_ds_dict, recall_topk, metric, log_dir)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='recall@1', factor=0.1, mode='max',
        patience=2, min_lr=1e-4, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint/' + config['model_name'],
        save_weights_only=False,
        monitor='recall@1',
        mode='max',
        save_best_only=True,
        verbose=1)
    # The callback logs should be synced.
    # This is a bug in tensorflow keras CallbackList class (v2.4.1).
    checkpoint._supports_tf_logs = False
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='recall@1',
        mode='max', patience=7,
        restore_best_weights=True)
    tensorboard_log = LogCallback(log_dir)

    callback_list.append(recall_eval)
    callback_list.append(checkpoint)
    callback_list.append(early_stop)
    if not config['lr_decay']:
        callback_list.append(reduce_lr)
    callback_list.append(tensorboard_log)
    if config['enable_prune']:
        callback_list += [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
            ]
    return callback_list, early_stop


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


def remove_subclassing_keras_model(model):
    return tf.keras.Model(model.inputs, model.outputs, name=model.name)


if __name__ == '__main__':
    config = train.config.config
    if config['mixed_precision']:
        print('---------------- Enabled Mixed Precision ----------------')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    train_ds, test_ds_dict = build_dataset(config)
    net = build_model(config)
    opt = build_optimizer(config)
    callbacks, early_stop = build_callbacks(config, test_ds_dict)
    net.compile(optimizer=opt)
    net.summary()
    try:
        net.fit(train_ds, epochs=config['epoch'], verbose=1,
            workers=input_pipeline.TF_AUTOTUNE,
            callbacks=callbacks)
    except KeyboardInterrupt as e:
        print('--')
        if early_stop.best_weights is None:
            print('Training canceled, but weights can not be restored because the best model is not available.')
        else:
            print('Training canceled and weights are restored from the best')
            net.set_weights(early_stop.best_weights)
    net = remove_subclassing_keras_model(net)
    if config['enable_prune']:
        net = tfmot.sparsity.keras.strip_pruning(net)
    net.save('{}.h5'.format(config['model_name']), include_optimizer=False)

    # print('Generating TensorBoard Projector for Embedding Vector...')
    # utils.visualize_embeddings(config)
