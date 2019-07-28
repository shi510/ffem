import json
import os
import importlib
import tensorflow as tf

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
def load_image_path_and_label(root_dir, list_file):
    images = []
    labels = []
    classes = {
        name : idx for idx, name in enumerate(os.listdir(root_dir))
    }

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
    return images, labels

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(img_path, label):
    image = tf.io.read_file(img_path)
    return preprocess_image(image), label

def get_model(model_type):
    as_file = model_type.rpartition('.')[0]
    as_fn = model_type.rpartition('.')[2]
    return getattr(importlib.import_module('model.' + as_file), as_fn)

def get_loss(loss_type):
    as_file = loss_type.rpartition('.')[0]
    as_fn = loss_type.rpartition('.')[2]
    return getattr(importlib.import_module('model.loss_fn.' + as_file), as_fn)
