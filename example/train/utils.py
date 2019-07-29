import json
import os
import importlib
import tensorflow as tf
import csv

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
