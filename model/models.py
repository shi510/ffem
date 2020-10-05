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


def get_model(name, shape):
    return model_list[name](shape)
