import json
import os
import importlib
import tensorflow as tf
import csv
import base64
import cv2
import struct

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


def read_RFW_train_list(path, race_list=["African", "Asian", "Caucasian", "Indian"], num_id=None):
    """
    The path is /your_root_path/BUPT-Balancedface/images/race_per_7000.
    It should contain thease folders below:
        African, Asian, Caucasian, Indian
    Returns a list that contains absolute path of all face images.
    """
    img_pathes = []
    img_labels = []
    unique_id = 0
    for race in race_list:
        race_path = os.path.join(path, race)
        num_id_list = os.listdir(race_path)
        if num_id is not None and len(num_id_list) > num_id:
            num_id_list = num_id_list[:num_id]
        for person_id in num_id_list:
            person_id_abs = os.path.join(race_path, person_id)
            if os.path.isdir(person_id_abs):
                file_names = os.listdir(person_id_abs)
                for item in file_names:
                    if item[-3:] == "jpg":
                        labeled = (os.path.join(race, person_id, item), unique_id)
                        img_pathes.append(labeled[0])
                        img_labels.append(labeled[1])
                unique_id += 1
    return img_pathes, img_labels, unique_id + 1