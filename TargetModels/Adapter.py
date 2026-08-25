import torch.nn as nn
import torch

class AdapterMLP(nn.Module):
    def __init__(self, input_shape, middle_dim=512, hidden_dim=1024):
        super().__init__()
        self.input_shape = input_shape
        flatten_input_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.middle_dim = middle_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_input_size, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.middle_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.hidden_dim//2, self.middle_dim)
        )
        
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.middle_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Linear(self.hidden_dim//2, self.hidden_dim),
            # nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, flatten_input_size),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x) 
        z_out = self.fc_encoder(x)

        z = self.fc_decoder(z_out)
        x_recon_flat = self.decoder(z)
        x_recon = x_recon_flat.view(-1, *self.input_shape)

        return x_recon, z_out
    
    def forward_from_mid(self, z):
        if self.decoder is None:
            raise RuntimeError("AdapterMLP not initialized. Run forward() first.")
        z = self.fc_decoder(z)
        x_recon_flat = self.decoder(z)
        x_recon = x_recon_flat.view(-1, *self.input_shape)
        
        return x_recon

class AdapterConv(nn.Module):
    def __init__(self, d = 64, middle_dim = 512):
        super().__init__()
        if d < 16:
            d = 16
        self.encoder = None
        self.decoder = None
        self.fc_encoder = None
        self.fc_decoder = None
        self.fc_input_size = None
        self.middle_dim = middle_dim
        self.d = d
        
    def build_encoder_decoder(self, x, device):
        B, C, H, W = x.shape

        self.encoder = nn.Sequential(
            nn.InstanceNorm2d(C, affine=True),
            nn.Conv2d(C, self.d//2, kernel_size = 3, padding= 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.d//2, self.d, kernel_size = 3, stride = 1, padding= 1),
            nn.LeakyReLU(0.2, inplace=True)
        ).to(device)

        output_padding_H = 1 if H % 2 == 0 else 0
        output_padding_W = 1 if W % 2 == 0 else 0
        output_padding = (output_padding_H, output_padding_W)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.d, self.d//2, kernel_size = 3, stride = 1, padding = 1, output_padding = 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.d//2, C, kernel_size = 3, stride = 1, padding = 1),
        ).to(device)


    def build_linear_layers(self, z, device):
        B, C, H, W = z.shape
        self.fc_input_size = (C, H, W)
        flatten_size = C * H * W
        self.fc_encoder = nn.Linear(flatten_size, self.middle_dim).to(device)
        self.fc_decoder = nn.Linear(self.middle_dim, flatten_size).to(device)


    def forward(self, x):
        if self.encoder is None:
            self.build_encoder_decoder(x, device=x.device)
        
        z = self.encoder(x)

        if self.fc_encoder is None:
            self.build_linear_layers(z, device=x.device)

        z = z.flatten(1)
        z_out = self.fc_encoder(z)
        # z_out = F.normalize(z_out, p=2, dim=1)
        h = self.fc_decoder(z_out)
        h = h.view(-1, *self.fc_input_size) 
        y = self.decoder(h)
        return y, z_out
    
    def forward_from_mid(self, z):
        assert self.fc_decoder is not None, "AdapterFC self.fc_decoder is None"
        h = self.fc_decoder(z)
        h = h.view(-1, *self.fc_input_size) 
        y = self.decoder(h)

        return y