import json
import os
import importlib
import tensorflow as tf
import csv
import base64
import struct
import json

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