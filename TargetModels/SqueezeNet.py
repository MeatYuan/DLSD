import torch
import torch.nn as nn
import torch.nn.init as init
class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        out1x1 = self.expand1x1(x)
        out3x3 = self.expand3x3(x)
        out = torch.cat([out1x1, out3x3], 1)
        out = self.expand_activation(out)
        return out
        
class SqueezeNet(nn.Module):
    def __init__(self, image_channels = 3, num_classes=10, dropout_rate=0.2,):
        super(SqueezeNet,self).__init__()
        self.features = nn.Sequential()
        self.features.append(nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.features.append(Fire(64, 16, 64, 64))
        self.features.append(Fire(128, 16, 64, 64))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        #更改
        self.features_sec = nn.Sequential()
        self.features_sec.append(Fire(128, 32, 128, 128))
        self.features_sec.append(Fire(256, 32, 128, 128))
        self.features_sec.append(nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True))  
        self.features_sec.append(Fire(256, 48, 192, 192))
        self.features_sec.append(Fire(384, 48, 192, 192))
        self.features_sec.append(Fire(384, 64, 256, 256))
        self.features_sec.append(Fire(512, 64, 256, 256))
        self.features_sec.append(nn.Dropout(p=dropout_rate))
        self.classifier = nn.Sequential()
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier.append(final_conv)
        self.classifier.append(nn.ReLU(inplace=True))
        self.classifier.append(nn.AdaptiveAvgPool2d((1, 1)))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        mid = x
        x = self.features_sec(x)
        feature = x.view(x.size(0), -1)
        x = self.classifier(x)
        y = torch.flatten(x, 1)
        output = {'logits':y, 'feature': feature,'Z': mid}
        return output
    
    def embed(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        feature = x.view(x.size(0), -1)

        return feature
    
    def forward_from_mid(self, z):
        x = self.features_sec(z)
        x = self.classifier(x)
        y = torch.flatten(x, 1)
        return {'logits':y}

    def forward_to_mid(self, x):
        x = self.features(x)
        return x
        
def get_squeezenet():
    return {
        "squeezenet": SqueezeNet,
    }
