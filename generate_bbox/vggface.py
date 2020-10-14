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
            img_path = line['NAME_ID'] + '.jpg'
            label = line['NAME_ID'].split('/')[0]
            x1 = int(line['X'])
            y1 = int(line['Y'])
            x2 = x1 + int(line['W'])
            y2 = y1 + int(line['H'])
            if label not in label_book:
                label_count += 1
                label_book[label] = label_count
                image_count[label] = 1
            else:
                image_count[label] += 1
            label = label_book[label]
            results[img_path] = {
                'label': label,
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            }
            if n % 10000 == 0:
                print('{} images are processed'.format(n))

    with open(out_file, 'w') as out_f:
        json.dump(results, out_f, indent=2)
