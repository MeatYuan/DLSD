from typing import Any, Callable
import torch
import torch.nn as nn
from torch import Tensor

#https://docs.pytorch.org/vision/main/_modules/torchvision/models/shufflenetv2.html
def channel_shuffle(x, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, num_channels, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats: list[int],
        stages_out_channels: list[int],
        image_channels: int = 3,
        num_classes: int = 10,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual, 
        
    ) -> None:
        super().__init__()
        
        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(image_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        image_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(image_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            image_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(image_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(output_channels, num_classes)
        self.features = nn.Sequential(
            self.conv1,
            self.maxpool,
            self.stage2,
        )
        self.features_sec = nn.Sequential(
            self.stage3,
            self.stage4,
            self.conv5
        )

    def forward(self, x) -> Tensor:
        # See note [TorchScript super()]
        x = self.features(x)
        x = self.features_sec(x)
        mid = x
        x = x.mean([2, 3])  # globalpool
        feature = x.view(x.size(0), -1)
        x = self.fc(x)
        output = {'logits':x, 'feature': feature,'Z': mid}
        return output

    def embed(self, x):
        # See note [TorchScript super()]
        x = self.features(x)
        x = self.features_sec(x)
        x = x.mean([2, 3])  # globalpool
        feature = x.view(x.size(0), -1)

        return feature
    
    def forward_from_mid(self, x):
        x = x.mean([2, 3])  # globalpool
        y = self.fc(x)
        output = {'logits': y}
        return output

    def forward_to_mid(self, x):
        x = self.features(x)
        x = self.features_sec(x)
        return x
        
def shufflenetv2(
    *args: Any,
    **kwargs: Any,
) -> ShuffleNetV2:

    model = ShuffleNetV2(stages_repeats = [4, 8, 4], stages_out_channels = [24, 116, 232, 464, 1024], *args, **kwargs)

    return model

def get_shufflenetv2():
    return {
        "shufflenetv2": shufflenetv2,
    }
