import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    cGAN Generator for 32x32 images.
    Official GeFL architecture: generators32/DCGAN.py (d=256 for CIFAR-10).
    Two-branch ConvTranspose: noise branch + label one-hot branch, then concat.
    """
    def __init__(self, output_channel=3, d=256):
        super().__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d * 2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d * 2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d * 2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, output_channel, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, z, y_label):
        # z: (B, 100, 1, 1), y_label: (B, 10, 1, 1)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(z)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(y_label)))
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        return torch.tanh(self.deconv4(x))

    def sample_image(self, args, sample_num):
        """
        Generate random-class images.
        Returns (images, one_hot_labels) matching official sample_image().
        one_hot_labels: float tensor (B, num_classes) on args.device.
        """
        with torch.no_grad():
            z = torch.randn(sample_num, 100, 1, 1).to(args.device)
            y = (torch.rand(sample_num, 1) * args.num_classes).long().view(-1)
            onehot = torch.zeros(args.num_classes, args.num_classes)
            onehot.scatter_(1, torch.arange(args.num_classes).view(-1, 1), 1)
            onehot = onehot.view(args.num_classes, args.num_classes, 1, 1)
            y_label = onehot[y].to(args.device)
            images = self.forward(z, y_label)
            # one-hot float labels (B, num_classes) — matches official one_hot(y_, num_classes)
            one_c = torch.zeros(sample_num, args.num_classes)
            for i, label in enumerate(y):
                one_c[i, label] = 1
        return images, one_c.to(args.device)


class Discriminator(nn.Module):
    """
    cGAN Discriminator for 32x32 images.
    Official GeFL architecture: generators32/DCGAN.py (d=64 for CIFAR-10).
    Two-branch Conv: image branch + label fill-map branch, then concat.
    Output: single validity scalar (no auxiliary classifier).
    """
    def __init__(self, img_channels=3, d=64):
        super().__init__()
        self.conv1_1 = nn.Conv2d(img_channels, d // 2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, d // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, img, y_fill):
        # img: (B, C, H, W), y_fill: (B, 10, H, W)
        x = F.leaky_relu(self.conv1_1(img), 0.2)
        y = F.leaky_relu(self.conv1_2(y_fill), 0.2)
        x = torch.cat([x, y], dim=1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        return torch.sigmoid(self.conv4(x))
