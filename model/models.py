import model.mobilenet_v2
import model.mobilenet_v3
import model.resnet
import model.logit_fn.margin_logit as margin_logit

import tensorflow as tf

model_list = {
    "MobileNetV2": model.mobilenet_v2.MobileNetV2,
    "MobileNetV3": model.mobilenet_v3.MakeMobileNetV3,
    "ResNet18": model.resnet.ResNet18
}


def attach_embedding(
    net,
    embeddings,
    kernel_regularizer=tf.keras.regularizers.l2(5e-4)
):
    out = net.output
    out = tf.keras.layers.Dense(embeddings,
    kernel_regularizer=kernel_regularizer,
        name='embedding')(out)
    return tf.keras.Model(net.input, out)


def attach_classifier(net, classes):
    """
    The output of the net should be embedding output.
    """
    out = net.output
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dense(classes, name='classifier')(out)
    return tf.keras.Model(net.input, out)


def attach_arc_margin_penalty(net, classes, logit_scale=30):
    """
    The output of the net should be embedding output.
    """
    out = net.output
    label = tf.keras.Input([])
    out = tf.keras.layers.BatchNormalization(name='pre_arc_margin')(out)
    out = margin_logit.ArcMarginPenaltyLogists(
        classes,
        logit_scale,
        name='arc_margin')(out, label)
    return tf.keras.Model([net.input, label], out)


def get_model(name, shape, dropout_before_avgpool=False, dropout_rate=0.2):
    net = model_list[name](shape)
    out = net.output
    if dropout_before_avgpool:
        out = tf.keras.layers.Dropout(rate=dropout_rate)(net.output)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Flatten(name='last_flatten')(out)
    return tf.keras.Model(net.input, out)
