import json

import numpy as np

def rect_from_landmark(lndmk):
    xs = np.zeros((5))
    ys = np.zeros((5))
    for i in range(5):
        xs[i], ys[i] = lndmk[i*2], lndmk[i*2+1]

    min_x = lndmk[np.argmin(xs)*2]
    min_y = lndmk[np.argmin(ys)*2+1]
    max_x = lndmk[np.argmax(xs)*2]
    max_y = lndmk[np.argmax(ys)*2+1]

    width = (max_x - min_x) / 1.5
    height = (max_y - min_y) / 1.5
    min_x -= width
    max_x += width
    min_y -= height
    max_y += height

    # Adjust the bounding box of yaw axis rotated face.
    left_eye = np.array([lndmk[0], lndmk[1]])
    right_eye = np.array([lndmk[2], lndmk[3]])
    nose = np.array([lndmk[4], lndmk[5]])
    left_line = np.linalg.norm(left_eye - nose)
    right_line = np.linalg.norm(right_eye - nose)
    shift = 0.
    if left_line - right_line > 0:
        shift = -(1 - right_line / left_line) * 1.2
    else:
        shift = (1 - left_line / right_line) * 1.2

    min_x += width * shift
    max_x += width * shift

    return np.array([min_x, min_y, max_x, max_y], dtype=np.int32)


def save_bbox_to_json(lndmk_file, out_file):
    f = open(lndmk_file, 'r')

    results = {}
    label_count = -1
    label_book = {}
    image_count = {}
    for n, line in enumerate(f):
        data = line.split(' ')
        img_path = data[0]
        label = data[1]
        lnd_mark = np.array([float(p) for p in data[2:]])
        bbox = rect_from_landmark(lnd_mark)
        if label not in label_book:
            label_count += 1
            label_book[label] = label_count
            image_count[label] = 1
        else:
            image_count[label] += 1
        label = label_book[label]
        # if label > 25000:
        #     break
        results[img_path] = {
            'label': label,
            'x1': int(bbox[0]), 'y1': int(bbox[1]),
            'x2': int(bbox[2]), 'y2': int(bbox[3])
        }
        if n % 10000 == 0:
            print('{} images are processed'.format(n))

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)