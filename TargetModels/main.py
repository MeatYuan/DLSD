import torch
from torch import nn, Tensor
from TargetModels.ResNet import get_resnet_models
from TargetModels.SqueezeNet import get_squeezenet
from TargetModels.VGG import get_vgg_model
from TargetModels.models import get_models
from TargetModels.MobileNetV2 import get_mobilenetv2
from TargetModels.MobileNetV3 import get_mobilenetv3
from TargetModels.ShuffleNetV2 import get_shufflenetv2

model_lists = [
    'cnn1layer', 'cnn2layer', 'cnn3layer', 'cnn4layer', 'lenet5', 'AlexNet', 'mobilenetv2', 'mobilenet_v3_small', 'shufflenetv2', 'squeezenet',
    'resnet8', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg13', 'vgg16', 'vgg19']
def get_target_models(args):
    all_models = {}
    all_models.update(get_models())
    all_models.update(get_mobilenetv2())
    all_models.update(get_mobilenetv3())
    all_models.update(get_shufflenetv2())
    all_models.update(get_squeezenet())
    all_models.update(get_resnet_models())
    all_models.update(get_vgg_model())
    
    models = []
    if args.Model_Heterogeneity:
        model_list = []
        for i in range(args.num_clients):
            if i % 3 == 0:
                model_list.append('AlexNet')
            elif i % 3 == 1:
                model_list.append('vgg11')
            else:
                model_list.append('ConvNet')

    else:
        model_list = ['AlexNet'] * args.num_clients



    for model_name in model_list[:args.num_clients]:
        if model_name not in all_models:
            raise ValueError(f"Model {model_name} is not defined in the available models.")
        elif model_name in model_lists[:5]:
            models.append(all_models[model_name](input_size=args.image_size, image_channels=args.image_channel, num_classes=args.num_classes))
        elif model_name == 'ConvNet':
            net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
            models.append(all_models[model_name](image_channels=args.image_channel, num_classes=args.num_classes,
                                                net_width=net_width, net_depth=net_depth,
                                                net_act=net_act, net_norm=net_norm,
                                                net_pooling=net_pooling, im_size=args.image_size))
        else:
            models.append(all_models[model_name](image_channels=args.image_channel, num_classes=args.num_classes))
    return models, model_list[:args.num_clients]

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

def get_model(model_name, args):
    all_models = {}
    all_models.update(get_models())
    all_models.update(get_mobilenetv2())
    all_models.update(get_mobilenetv3())
    all_models.update(get_shufflenetv2())
    all_models.update(get_squeezenet())
    all_models.update(get_resnet_models())
    all_models.update(get_vgg_model())

    if model_name not in all_models:
        raise ValueError(f"Model {model_name} is not defined in the available models.")
    elif model_name in model_lists[:5]:
        return all_models[model_name](input_size=args.image_size, image_channels=args.image_channel, num_classes=args.num_classes)
    elif model_name == 'ConvNet':
        net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
        return all_models[model_name](image_channels=args.image_channel, num_classes=args.num_classes,
                                                net_width=net_width, net_depth=net_depth,
                                                net_act=net_act, net_norm=net_norm,
                                                net_pooling=net_pooling, im_size=args.image_size)
    else:
        return all_models[model_name](image_channels=args.image_channel, num_classes=args.num_classes)