import tensorflow as tf
import tensorflow.keras.layers as layers

# 224, 192, 160
def RestNet18(shape):
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
    for n in range(2):
        shortcut = input
        x = _conv_bn_act(input, 64, 3, 1, 'same', layers.ReLU())
        x = _conv_bn_act(x, 64, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
    return x

def _res_block2(input):
    shortcut = layers.Conv2D(128, 2, 2, 'valid', use_bias=False)(input)
    x = _conv_bn_act(input, 128, 3, 2, 'same', layers.ReLU())
    x = _conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
    x = layers.Add()([x, shortcut])
    for n in range(2):
        shortcut = x
        x = _conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
        x = _conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
    return x

def _res_block3(input):
    shortcut = layers.Conv2D(256, 2, 2, 'valid', use_bias=False)(input)
    x = _conv_bn_act(input, 256, 3, 2, 'same', layers.ReLU())
    x = _conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
    x = layers.Add()([x, shortcut])
    for n in range(2):
        shortcut = x
        x = _conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
        x = _conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
    return x

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

"""
It does not work.

model = ResNet34Experiment()
model.build((64, 160, 160, 3))
model.summery()

It can not build the model.
"""
class ResNet34Experiment(tf.keras.Model):

    def __init__(self):
        super(ResNet34Experiment, self).__init__()

    def call(self, inputs):
        x = self._conv_bn_act(inputs, 32, 7, 2, 'same', layers.ReLU())
        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = self._res_block1(x)
        x = self._res_block2(x)
        x = self._res_block3(x)
        x = self._res_block4(x)
        x = layers.AveragePooling2D((x.shape[1],x.shape[2]))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        return x

    def _res_block1(self, input):
        for n in range(3):
            shortcut = input
            x = self._conv_bn_act(input, 32, 3, 1, 'same', layers.ReLU())
            x = self._conv_bn_act(x, 32, 3, 1, 'same', layers.ReLU())
            x = layers.Add()([x, shortcut])
        return x

    def _res_block2(self, input):
        shortcut = layers.Conv2D(64, 1, 2, 'valid')(input)
        x = self._conv_bn_act(input, 64, 3, 2, 'same', layers.ReLU())
        x = self._conv_bn_act(x, 64, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
        for n in range(3):
            shortcut = x
            x = self._conv_bn_act(x, 64, 3, 1, 'same', layers.ReLU())
            x = self._conv_bn_act(x, 64, 3, 1, 'same', layers.ReLU())
            x = layers.Add()([x, shortcut])
        return x

    def _res_block3(self, input):
        shortcut = layers.Conv2D(128, 1, 2, 'valid')(input)
        x = self._conv_bn_act(input, 128, 3, 2, 'same', layers.ReLU())
        x = self._conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
        for n in range(5):
            shortcut = x
            x = self._conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
            x = self._conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
            x = layers.Add()([x, shortcut])
        return x

    def _res_block4(self, input):
        shortcut = layers.Conv2D(256, 1, 2, 'valid')(input)
        x = self._conv_bn_act(input, 256, 3, 2, 'same', layers.ReLU())
        x = self._conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
        for n in range(2):
            shortcut = x
            x = self._conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
            x = self._conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
            x = layers.Add()([x, shortcut])
        return x

    def _conv_bn_act(self, input, filter, ksize, stride, padding, act):
        x = layers.Conv2D(
            filter,
            (ksize, ksize),
            stride,
            padding,
            use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = act(x)
        return x

"""
It does not work.

model = ResNet18Experiment()
model.build((64, 160, 160, 3))
model.summery()

It can not build the model.
"""
class ResNet18Experiment(tf.keras.Model):

    def __init__(self):
        super(ResNet18Experiment, self).__init__()

    def call(self, inputs):
        x = self._conv_bn_act(inputs, 64, 7, 2, 'same', layers.ReLU())
        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = self._res_block1(x)
        x = self._res_block2(x)
        x = self._res_block3(x)
        avgp = layers.GlobalAveragePooling2D()(x)
        # maxp = layers.MaxPool2D(pool_size=x.shape[1])(x)
        # concat = layers.Concatenate()([avgp, maxp])
        concat = layers.Flatten()(avgp)
        feature = layers.Dense(128)(concat)
        return feature

    def _res_block1(self, input):
        for n in range(2):
            shortcut = input
            x = self._conv_bn_act(input, 64, 3, 1, 'same', layers.ReLU())
            x = self._conv_bn_act(x, 64, 3, 1, 'same', layers.ReLU())
            x = layers.Add()([x, shortcut])
        return x

    def _res_block2(self, input):
        shortcut = layers.Conv2D(128, 2, 2, 'valid', use_bias=False)(input)
        x = self._conv_bn_act(input, 128, 3, 2, 'same', layers.ReLU())
        x = self._conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
        for n in range(2):
            shortcut = x
            x = self._conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
            x = self._conv_bn_act(x, 128, 3, 1, 'same', layers.ReLU())
            x = layers.Add()([x, shortcut])
        return x

    def _res_block3(self, input):
        shortcut = layers.Conv2D(256, 2, 2, 'valid', use_bias=False)(input)
        x = self._conv_bn_act(input, 256, 3, 2, 'same', layers.ReLU())
        x = self._conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
        x = layers.Add()([x, shortcut])
        for n in range(2):
            shortcut = x
            x = self._conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
            x = self._conv_bn_act(x, 256, 3, 1, 'same', layers.ReLU())
            x = layers.Add()([x, shortcut])
        return x

    def _conv_bn_act(self, input, filter, ksize, stride, padding, act):
        x = layers.Conv2D(
            filter,
            (ksize, ksize),
            stride,
            padding,
            use_bias=False)(input)
        x = layers.BatchNormalization()(x)
        x = act(x)
        return x
