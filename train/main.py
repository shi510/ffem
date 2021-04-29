import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import train.input_pipeline as input_pipeline
import train.config
from train.callbacks import LogCallback
from train.callbacks import RecallCallback
from train.utils import apply_pruning
from train.utils import apply_quantization_aware
import net_arch.models
import train.blocks
from train.custom_models.softmax_center_model import SoftmaxCenterModel
from train.custom_models.angular_margin_model import AngularMarginModel
from train.custom_models.group_aware_model import GroupAwareModel

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.mixed_precision as mixed_precision
import tensorflow_model_optimization as tfmot
import numpy as np


def build_dataset(config):
    train_ds, test_ds_dict = input_pipeline.make_tfdataset(
        config['train_file'],
        config['test_files'],
        config['batch_size'],
        config['shape'][:2])
    return train_ds, test_ds_dict


def build_backbone_model(config):
    is_pretrained = False
    if os.path.exists(config['saved_backbone']):
        net = tf.keras.models.load_model(config['saved_backbone'])
        print('\n---------------- Restore Backbone Network ----------------\n')
        print(config['saved_backbone'])
        print('\n----------------------------------------------------------\n')
        is_pretrained = True
    else :
        net = net_arch.models.get_model(config['model'], config['shape'])

    return net, is_pretrained


def build_model(config):
    net, is_pretrained = build_backbone_model(config)
    if config['enable_prune']:
        net = apply_pruning(
            net, config['prune_params'], 1 if is_pretrained else None)

    param = copy.deepcopy(config['loss_param'][config['loss']])
    dummy_x = np.zeros([config['batch_size']] + config['shape'])
    dummy_y = np.zeros([config['batch_size']]+[config['num_identity']])
    model = None
    if config['loss'] == 'SoftmaxCenter':
        param['n_classes'] = config['num_identity']
        param['embedding_dim'] = config['embedding_dim']
        model = SoftmaxCenterModel(net, **param, name=config['model_name'])
    elif config['loss'] == 'AngularMargin':
        param['n_classes'] = config['num_identity']
        param['embedding_dim'] = config['embedding_dim']
        model = AngularMarginModel(net, **param, name=config['model_name'])
    elif config['loss'] == 'GroupAware':
        param['n_classes'] = config['num_identity']
        param['instance_dim'] = config['embedding_dim']
        model = GroupAwareModel(net, **param, name=config['model_name'])
    else:
        raise Exception('The loss ({}) is not supported.'.format(config['loss']))

    restore_latest_checkpoint(model, config['checkpoint'])
    if config['enable_quant_aware']:
        model.backbone = apply_quantization_aware(model.backbone, None)
    model([dummy_x, dummy_y], training=True) # dummy call for building model
    return model


def build_callbacks(config, test_ds_dict):
    log_dir = os.path.join('logs', config['model_name'])
    callback_list = []
    metric = config['eval']['metric']
    recall_topk = config['eval']['recall']
    recall_eval = RecallCallback(test_ds_dict, recall_topk, metric, log_dir)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='recall@1', factor=0.5, mode='max',
        patience=2, min_lr=1e-4, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/{}/best'.format(config['model_name']),
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
            tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)]
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


def restore_latest_checkpoint(net, checkpoint_path):
    checkpoint = tf.train.Checkpoint(net)
    latest_path = tf.train.latest_checkpoint(checkpoint_path)
    print('\n---------------- Restore Checkpoint ----------------\n')
    if latest_path is not None:
        print('restore_latest_checkpoint:', latest_path)
        checkpoint.restore(latest_path).expect_partial()
    else:
        print('Can not find latest checkpoint file:', checkpoint_path)
    print('\n----------------------------------------------------\n')

def start_training(config):
    if config['mixed_precision']:
        print('---------------- Enabled Mixed Precision ----------------')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    train_ds, test_ds_dict = build_dataset(config)
    train_net = build_model(config)
    opt = build_optimizer(config)
    callbacks, early_stop = build_callbacks(config, test_ds_dict)
    train_net.compile(optimizer=opt)
    train_net.summary()
    try:
        train_net.fit(train_ds, epochs=config['epoch'], verbose=1,
            workers=input_pipeline.TF_AUTOTUNE,
            callbacks=callbacks)
    except KeyboardInterrupt:
        print('--')
        if early_stop.best_weights is None:
            print('Training is canceled, but weights can not be restored because the best model is not available.')
        else:
            print('Training is canceled and weights are restored from the best')
            train_net.set_weights(early_stop.best_weights)

    if config['enable_prune']:
        train_net = tfmot.sparsity.keras.strip_pruning(train_net)
    infer_model = train_net.get_inference_model()
    infer_model.save('{}.h5'.format(infer_model.name))
    train_net.backbone.save('{}_backbone.h5'.format(train_net.name))

if __name__ == '__main__':
    config = train.config.config
    start_training(config)
