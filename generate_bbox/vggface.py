import csv
import json


def save_bbox_to_json(lndmk_file, out_file):
    results = {}
    label_count = -1
    label_book = {}
    image_count = {}
    with open(lndmk_file, 'r') as lndmk_f:
        reader = csv.DictReader(lndmk_f, delimiter=',')
        for n, line in enumerate(reader):
            img_path = line['NAME_ID']
            label = line['NAME_ID'].split('/')[0]
            bbox = [line['X'], line['Y'], line['W'], line['H']]
            if label not in label_book:
                label_count += 1
                label_book[label] = label_count
                image_count[label] = 1
            else:
                image_count[label] += 1
            label = label_book[label]
            results[img_path] = {
                'label': label,
                'x1': int(bbox[0]), 'y1': int(bbox[1]),
                'x2': int(bbox[2]), 'y2': int(bbox[3])
            }
            if n % 10000 == 0:
                print('{} images are processed'.format(n))

    with open(out_file, 'w') as out_f:
        json.dump(results, out_f, indent=2)
