import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models._utils import _make_divisible
from typing import Any, Callable, List, Optional

#https://docs.pytorch.org/vision/main/_modules/torchvision/models/mobilenetv2.html#mobilenet_v2
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(
        self,
        image_channels = 3,
        num_classes: int = 10,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[list[list[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if block is None:
            block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [Conv2dNormActivation(image_channels, input_channel, stride=1, norm_layer=norm_layer)]

        for t, c, n, s in inverted_residual_setting[:2]:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)
        # 更改
        features_sec =[]
        for t, c, n, s in inverted_residual_setting[2:]:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features_sec.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel    
        features_sec.append(Conv2dNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features_sec = nn.Sequential(*features_sec)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x) -> Tensor:
        x = self.features(x)
        x = self.features_sec(x)
        mid = x
        x = self.pool(x)
        feature = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = {'logits':x, 'feature': feature,'Z': mid}
        return output

    def embed(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        x = self.pool(x)
        feature = x.view(x.size(0), -1)

        return feature
    
    def forward_from_mid(self, x):
        #x = self.features_sec(x)
        x = self.pool(x)
        y = self.classifier(x)
        return {'logits': y}

    def forward_to_mid(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        return x

def mobilenet_v2(**kwargs: Any) -> MobileNetV2:
    """MobileNetV2 architecture from the paper
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    """
    model = MobileNetV2(**kwargs)
    return model

def get_mobilenetv2():
    return {
        "mobilenetv2": mobilenet_v2,
    }