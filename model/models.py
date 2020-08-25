import model.mobilenet_v2
import model.resnet

model_list = {
    "MobileNetV2": model.mobilenet_v2.MobileNetV2,
    "ResNet18": model.resnet.ResNet18
}

def get_model(name, shape):
    return model_list[name](shape)