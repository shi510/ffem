import tensorflow as tf
import tensorflow.keras.layers as layers

# 224, 192, 160
def ResNet18(shape):
    input = tf.keras.Input(shape)
    x = _conv_bn_act(input, 64, 7, 2, 'same', layers.ReLU())
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = _res_block1(x)
    x = _res_block2(x)
    x = _res_block3(x)
    avgp = layers.GlobalAveragePooling2D()(x)
    # maxp = layers.MaxPool2D(pool_size=x.shape[1])(x)
    # concat = layers.Concatenate()([avgp, maxp])
    concat = layers.Flatten()(avgp)
    feature = layers.Dense(128)(concat)
    return tf.keras.Model(input, feature)

def _res_block1(input):
    x = input
    for _ in range(2):
        shortcut = x
        x = _conv_bn_act(x, 64, 3, 1, 'same', layers.ReLU())
        x = _conv_bn_act(x, 64, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
    return x

def _res_block2(input):
    shortcut = _res_downsample(128)(input)
    x = _conv_bn_act(input, 128, 3, 2, 'same', layers.ReLU())
    x = _conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
    x = layers.Add()([x, shortcut])
    for _ in range(2):
        shortcut = x
        x = _conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
        x = _conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
    return x

def _res_block3(input):
    shortcut = _res_downsample(256)(input)
    x = _conv_bn_act(input, 256, 3, 2, 'same', layers.ReLU())
    x = _conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
    x = layers.Add()([x, shortcut])
    for _ in range(2):
        shortcut = x
        x = _conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
        x = _conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
    return x

def _res_downsample(filter):
    return layers.Conv2D(filter, 1, 2, 'valid', use_bias=False)

def _conv_bn_act(input, filter, ksize, stride, padding, act):
    x = layers.Conv2D(
        filter,
        (ksize, ksize),
        stride,
        padding,
        use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = act(x)
    return x
