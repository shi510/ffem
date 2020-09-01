import math
import os
import tensorflow as tf
import numpy as np
import example.train.utils as utils
import example.train.input_pipeline as input_pipeline
import example.train.config
import model.models
from tensorboard.plugins import projector


def build_dataset(config):
    if config['dataset'] == "RFW":
        rfw_config = config['__{}'.format("RFW")]
        ds, num_class = input_pipeline.make_RFW_tfdataset(
            root_path=rfw_config['train_path'],
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
    net = None
    loss_fn = None

    if os.path.exists(config['saved_model']):
        net = tf.keras.models.load_model(
            config['saved_model'],
            custom_objects={
                'softmax_cross_entropy_with_logits_v2':
                    tf.nn.softmax_cross_entropy_with_logits,
                'original_triplet_loss':
                    utils.get_loss('triplet_loss.original_triplet_loss'),
                'adversarial_triplet_loss':
                    utils.get_loss('triplet_loss.adversarial_triplet_loss'),
            })
        if not config['train_classifier']:
            net = tf.keras.Model(net.input, net.get_layer('flatten').output)
            net.trainable = False
            embedding = tf.keras.layers.Dense(
                config['embedding_dim'], use_bias=False)(net.output)
            net = tf.keras.Model(net.input, embedding)
    else :
        net = model.models.get_model(config['model'], config['shape'])
        embedding = tf.keras.layers.GlobalAveragePooling2D()(net.output)
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

        net = tf.keras.Model(net.input, embedding)

    if config['train_classifier']:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits
    else:
        loss_fn = utils.get_loss(config['metric_loss'])
    return net, loss_fn


def build_callbacks():
    callback_list = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5,
        patience=5, min_lr=1e-5)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint',
        save_weights_only=False,
        monitor='loss',
        mode='max',
        save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    callback_list.append(reduce_lr)
    callback_list.append(checkpoint)
    callback_list.append(early_stop)
    return callback_list


def visualize_embeddings(config):
    path = config['__RFW']['train_path']
    race_list = config['__RFW']['race_list']
    img_pathes, labels, _ = utils.read_RFW_train_list(path, race_list=race_list, num_id=20)
    model = tf.keras.models.load_model('{}.h5'.format(config['model_name']),
        custom_objects={
            'softmax_cross_entropy_with_logits_v2':
                tf.nn.softmax_cross_entropy_with_logits,
            'original_triplet_loss':
                utils.get_loss('triplet_loss.original_triplet_loss'),
            'adversarial_triplet_loss':
                utils.get_loss('triplet_loss.adversarial_triplet_loss'),
        })
    if config['train_classifier']:
        last_out = model.get_layer('flatten').output
        embedding_dim = last_out.shape[-1]
    else:
        last_out = model.output
        embedding_dim = config['embedding_dim']
    embeddings = tf.math.l2_normalize(last_out, axis=1)
    model = tf.keras.Model(model.input, embeddings)
    embedding_array = np.ndarray((len(img_pathes), embedding_dim))

    for n, img_path in enumerate(img_pathes):
        img_path = os.path.join(path, img_path)
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
        for id in labels:
            f.write("{}\n".format(id))

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


if __name__ == '__main__':
    config = example.train.config.config
    train_ds, num_class = build_dataset(config)
    model, loss_fn = build_model(config, num_class)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    if config['use_keras']:
        model.compile(optimizer=opt, loss=loss_fn)
        model.fit(train_ds, epochs=config['epoch'], verbose=1,
            workers=10, callbacks=build_callbacks())
        model.save('{}.h5'.format(config['model_name']))
    else:
        # Iterate over epochs.
        for epoch in range(config['epoch']):
            print('Start of epoch %d' % epoch)
            # Iterate over the batches of the dataset.
            epoch_loss = np.zeros(1, dtype=np.float32)
            print(model.trainable_weights)
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