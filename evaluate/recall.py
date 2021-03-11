from train.utils import pairwise_distance

import numpy as np
import tensorflow as tf


def calc_top_k_label(dist, label, top_k, largest=True):
    if not largest:
        dist = -1 * dist
    
    top_k_label = []
    for x in dist:
        indices = tf.math.top_k(x, top_k)[1]
        top_k_label.append([label[i] for i in indices])

    return top_k_label


def evaluate(model, dataset, metric, top_k: list, batch_size=256, norm=True):
    if metric =='cos':
        metric_fn = lambda X, Y: tf.matmul(X, tf.transpose(Y))
        largest = True
    elif metric == 'l2':
        metric_fn = lambda X, Y: pairwise_distance(X, Y)
        largest = False
    else:
        raise 'Unsupported metric.'

    X = []
    Y = []
    # extract all embeddings in dataset.
    for batch_x, batch_y in dataset:
        batch_pred = model(batch_x)
        if norm:
            batch_pred = tf.math.l2_normalize(batch_pred, axis=1)
        X.append(batch_pred)
        Y.append(batch_y)
    X = tf.concat(X, axis=0)
    Y = tf.concat(Y, axis=0)
    ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.batch(batch_size)
    P = []
    max_top_k = np.max(top_k)
    # calculate top_k using batched sample for memory efficiency.
    for n, batch_x in enumerate(ds):
        dist = metric_fn(batch_x, X)
        # remove self distance.
        max_dist = float('-inf') if largest else float('inf')
        max_dist = tf.fill(batch_x.shape[0], max_dist)
        dist = tf.linalg.set_diag(dist, max_dist, k=n*batch_x.shape[0])
        # get top_k label
        batch_top_k_label = calc_top_k_label(dist, Y, max_top_k, largest)
        # merge list
        P = P + batch_top_k_label

    top_k_results = []
    for k in top_k:
        s = sum([1 for y, p in zip(Y, P) if y in p[:k]])
        top_k_results.append(s / len(Y))

    return top_k_results
