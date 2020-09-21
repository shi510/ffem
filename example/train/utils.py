import base64
import csv
import importlib
import json
import os
import struct

import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np

import model.loss_fn.default_loss as default_loss


def open_config_file(file_path):
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file

# the each path in list_path is relative path.
# /usr/ml/faces/train_list.txt
# in train_list.txt
#     n0002/1.jpg
#     n0002/2.jpg
#     ...
#     n1234/1.jpg
# the train classes(ex, n0002) are in /usr/ml/faces/train.
# it should be converted to absolute path.
# for example, os.path.join('/usr/ml/faces/train', 'n0002/1.jpg')
# the root_dir is '/usr/ml/faces/train'
def load_image_path_and_label(root_dir, list_file, crop_list):
    images = []
    labels = []
    crops = []
    n = 0
    classes = {
        name : idx for idx, name in enumerate(os.listdir(root_dir))
    }
    crop_file = open(crop_list, 'r')
    crop_boxes = csv.reader(crop_file)
    crop_boxes = list(crop_boxes)
    crop_file.close()

    with open(list_file, 'r') as f:
        while(True):
            img = f.readline()
            if img == "":
                break
            label = classes[img.split('/')[0]]
            img = os.path.join(root_dir, img)
            if '\n' in img[-1]:
                img = img[0:-1]
            images.append(img)
            labels.append(label)
            box = crop_boxes[n + 1][1:5]
            crops.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            n += 1
    return images, labels, crops

def get_model(model_type):
    as_file = model_type.rpartition('.')[0]
    as_fn = model_type.rpartition('.')[2]
    return getattr(importlib.import_module('model.' + as_file), as_fn)

def get_loss(loss_type):
    as_file = loss_type.rpartition('.')[0]
    as_fn = loss_type.rpartition('.')[2]
    return getattr(importlib.import_module('model.loss_fn.' + as_file), as_fn)

def read_ms_1m_celeb(file_path):
    with open(file_path, 'r') as tsvF:
        reader = csv.reader(tsvF, delimiter='\t')
        i = 0
        for row in reader:
            MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])
            print(row[:-1], base64.b64decode(row[5]))
            save_path = '/home/shi510/sh_dev_dir/{}/{}_{}.jpg'.format(MID, imgSearchRank, faceID)
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))
                input()
            with open(save_path, 'wb') as f:
                f.write(data)


def read_dataset_from_json(list_file):
    """
    Input:
        list_file: it should be saved with the format (json) as below.  
        {
        "Asian/m.0hn95h9/000012_00@en.jpg": {
            "label": 0,
            "x1": 9,
            "y1": 13,
            "x2": 75,
            "y2": 100
        },
        "Asian/m.0hn95h9/000046_00@ja.jpg": {
            "label": 0,
            "x1": 7,
            "y1": 7,
            "x2": 43,
            "y2": 65
        },
        ...
        }

    Return pathes, labels, boxes
    """
    pathes = []
    labels = []
    boxes = []
    with open(list_file, 'r') as f:
        dataset = json.loads(f.read())
    for file_name in dataset:
        content = dataset[file_name]
        pathes.append(file_name)
        labels.append(content['label'])
        box = [float(content['y1']), float(content['x1']),
            float(content['y2']), float(content['x2'])]
        boxes.append(box)

    return pathes, labels, boxes


def visualize_embeddings(config, identities=50):
    pathes, labels, boxes = read_dataset_from_json(config['train_file'])
    indexed_data = []
    for n, (path, label, box) in enumerate(zip(pathes, labels, boxes)):
        if label >= identities:
            break
        indexed_data.append({'path': path, 'label': label, 'box': box})
    model = tf.keras.models.load_model(config['saved_model'],
        custom_objects=keras_model_custom_obj)
    if os.path.isdir(config['saved_model']):
        model.load_weights(config['saved_model']+'/variables/variables')
    if config['train_classifier']:
        last_out = model.get_layer('embedding').output
        embedding_dim = last_out.shape[-1]
    else:
        last_out = model.output
        embedding_dim = config['embedding_dim']
    # embeddings = tf.math.l2_normalize(last_out, axis=1, name='l2_norm')
    embeddings = last_out
    model = tf.keras.Model(model.input, embeddings)
    embedding_array = np.ndarray((len(indexed_data), embedding_dim))

    for n, attr in enumerate(indexed_data):
        img_path = os.path.join(config['img_root_path'], attr['path'])
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        shape = tf.shape(img)
        shape = tf.repeat(shape, [2, 2, 0])
        shape = tf.scatter_nd([[0], [2], [1], [3]], shape, tf.constant([4]))
        box = np.array(attr['box'], dtype=np.float32)
        box /= tf.cast(shape, tf.float32)
        box = tf.clip_by_value(box, 0, 1)
        img = tf.image.crop_and_resize([img], [box], [0], config['shape'][:2])[0]
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


keras_model_custom_obj = {
    'softmax_xent_loss_wrap':
        default_loss.softmax_xent_loss_wrap,
    'original_triplet_loss':
        get_loss('triplet_loss.original_triplet_loss'),
    'adversarial_triplet_loss':
        get_loss('triplet_loss.adversarial_triplet_loss'),
}
