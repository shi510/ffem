import tensorflow as tf

def EfficientNetB3(shape):
    model = tf.keras.applications.EfficientNetB3(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights=None)
    return model