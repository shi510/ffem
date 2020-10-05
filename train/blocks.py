import tensorflow as tf

import train.layers
import model.logit_fn.margin_logit as margin_logit


def attach_arc_margin_penalty(
    x : tf.Tensor,
    label : tf.Tensor,
    classes : int,
    scale=30):
    out = x
    out = margin_logit.ArcMarginPenaltyLogists(
        classes,
        scale,
        name='arc_margin')(out, label)
    return out


def attach_embedding_projection(x : tf.Tensor, embedding_dim : int):
    out = x
    out = tf.keras.layers.Dense(embedding_dim,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4))(out)
    out = tf.keras.layers.BatchNormalization(name='embeddings')(out)
    return out
