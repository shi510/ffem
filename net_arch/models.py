import net_arch.mobilenet_v2
import net_arch.mobilenet_v3
import net_arch.resnet
import net_arch.logit_fn.margin_logit as margin_logit

import tensorflow as tf

model_list = {
    "MobileNetV2": net_arch.mobilenet_v2.MobileNetV2,
    "MobileNetV3": net_arch.mobilenet_v3.MakeMobileNetV3,
    "ResNet18": net_arch.resnet.ResNet18
}


def get_model(name, shape):
    return model_list[name](shape)
