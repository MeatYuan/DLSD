import torch
import torch.nn as nn
from typing import cast

cfgs = {
    11 : [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    13 : [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    16 : [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    19 : [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],}
#https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg11
def make_layers(cfg, in_channels = 3, batch_norm=False):
    layers = []
    layers_sec = []
    M_POS = [i+1 for i, v in enumerate(cfg) if v == "M"]
    for v in cfg[:M_POS[1]]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    for v in cfg[M_POS[1]:]:
        if v == "M":
            layers_sec += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers_sec += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers_sec += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return layers, layers_sec

class VGG(nn.Module):
    def __init__(self, cfg, image_channels = 3, num_classes = 10, batch_norm = True, init_weights = True, dropout_rate = 0.2,):
        super(VGG, self).__init__()
        layers, layers_sec = make_layers(cfg, in_channels=image_channels, batch_norm=batch_norm)
        self.features = nn.Sequential(*layers)
        self.features_sec = nn.Sequential(*layers_sec)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        mid = x
        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        output = {'logits':x, 'feature': feature,'Z': mid}
        return output

    def embed(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)

        return feature
    
    def forward_from_mid(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return {'logits':y}

    def forward_to_mid(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        return x
        
def vgg11(image_channels=3, num_classes=10, **kwargs):
    """Constructs a VGG-11 model."""
    return VGG(cfg = cfgs[11], image_channels=image_channels, num_classes=num_classes, **kwargs)

def vgg13(image_channels=3, num_classes=10, **kwargs):
    """Constructs a VGG-13 model."""
    return VGG(cfg = cfgs[13],image_channels=image_channels, num_classes=num_classes, **kwargs)

def vgg16(image_channels=3, num_classes=10, **kwargs):
    """Constructs a VGG-16 model."""
    return VGG(cfg = cfgs[16],image_channels=image_channels, num_classes=num_classes, **kwargs)

def vgg19(image_channels=3, num_classes=10, **kwargs):
    """Constructs a VGG-19 model."""
    return VGG(cfg = cfgs[19],image_channels=image_channels,  num_classes=num_classes, **kwargs)

def get_vgg_model():
    return {
        "vgg11": vgg11,
        "vgg13": vgg13,
        "vgg16": vgg16,
        "vgg19": vgg19
    }

# def Vgg(in_channels=3, VGG_type=11, num_classes=10, **kwargs):
#     """Constructs a VGG model with specified type."""
#     if VGG_type == 11:
#         return vgg11(in_channels=in_channels, num_classes=num_classes, **kwargs)
#     elif VGG_type == 13:
#         return vgg13(in_channels=in_channels, num_classes=num_classes, **kwargs)
#     elif VGG_type == 16:
#         return vgg16(in_channels=in_channels, num_classes=num_classes, **kwargs)
#     elif VGG_type == 19:
#         return vgg19(in_channels=in_channels, num_classes=num_classes, **kwargs)
#     else:
#         raise ValueError(f"Unsupported VGG type: {VGG_type}")
