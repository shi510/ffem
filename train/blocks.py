import tensorflow as tf


def attach_GNAP(x : tf.Tensor):
    out = x
    norm = tf.norm(out, ord=2, axis=3, keepdims=True)
    norm = tf.math.maximum(norm, 1e-12)
    mean = tf.math.reduce_mean(norm)
    out = tf.keras.layers.BatchNormalization(scale=False)(out)
    out = tf.math.divide(out, norm)
    out = tf.multiply(out, mean)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.BatchNormalization(scale=False)(out)
    return out


def attach_l2_norm_features(x : tf.Tensor, scale=30):
    x = tf.math.l2_normalize(x, axis=1)
    x = tf.multiply(x, scale)
    return x


def attach_embedding_projection(x : tf.Tensor, embedding_dim : int):
    out = x
    out = tf.keras.layers.Dense(embedding_dim,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4))(out)
    out = tf.keras.layers.BatchNormalization(name='embeddings')(out)
    return out
