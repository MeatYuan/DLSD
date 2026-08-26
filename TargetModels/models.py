import torch
import torch.nn as nn 
from torch import Tensor
#-------------- AlexNet Models---------------
class AlexNet(nn.Module):
    def __init__(self, image_channels=1, num_classes=10, dropout_rate=0.2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 128, kernel_size=5, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),)
        
        self.features_sec = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride = 2))
        
        self.classifier = nn.Linear(192 * 4 * 4, num_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(4096, num_classes))

    def forward(self, x) -> Tensor:
        x = self.features(x)
        x = self.features_sec(x)
        mid = x 
        feature = x.view(x.size(0), -1)
        x = self.classifier(feature)
        output = {'logits':x, 'feature': feature,'Z': mid}
        return output

    def embed(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        feature = x.view(x.size(0), -1)

        return feature
    
    def forward_from_mid(self, x: Tensor):
        # x = self.features_sec(x)
        feature = x.view(x.size(0), -1)
        y = self.classifier(feature)
        return {'logits' : y}

    def forward_to_mid(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        return x
        
    
class ConvNet(nn.Module):
    def __init__(self, image_channels, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, self.features_sec, shape_feat = self._make_layers(image_channels, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        mid = x
        feature = x.view(x.size(0), -1)
        # print(out.size())
        # print(self.classifier)
        out = self.classifier(feature)
        output = {'logits':out, 'feature': feature,'Z': mid}
        return output

    def embed(self, x):
        out = self.features(x)
        out = self.features_sec(out)
        out = out.view(out.size(0), -1)
        return out

    def forward_from_mid(self, x: Tensor):
        # x = self.features_sec(x)
        feature = x.view(x.size(0), -1)
        y = self.classifier(feature)
        return {'logits' : y}
    
    def forward_to_mid(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        return x
        
    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        # if im_size[0] == 28:
        #     im_size = (32, 32)
        split_point = net_depth // 2 
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(split_point):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        feature_one = nn.Sequential(*layers)
        layers = []
        for d in range(split_point, net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        feature_two = nn.Sequential(*layers)

        return feature_one, feature_two, shape_feat

class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)
    
def get_models():
    return {
        "AlexNet": AlexNet,
        "ConvNet": ConvNet
    }