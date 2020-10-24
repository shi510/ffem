import tensorflow as tf


def softmax_xent_loss_wrap(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    return tf.reduce_mean(loss)
