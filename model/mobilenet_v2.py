import tensorflow as tf

def MobileNetV2(shape, name='MobileNetV2'):
    model = tf.keras.applications.MobileNetV2(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights=None)
    return model