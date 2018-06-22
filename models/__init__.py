from models import resnet

archs = {
    'resnet50': resnet.ResNet50,
    'resnet101': resnet.ResNet101,
    'resnet152': resnet.ResNet152,
}
