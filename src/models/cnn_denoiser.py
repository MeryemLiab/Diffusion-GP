import torch
import torch.nn as nn


class CNNDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_t, t):
        b, _, h, w = x_t.shape

        t = t.view(b, 1, 1, 1)
        t_channel = t.expand(b, 1, h, w)

        x = torch.cat([x_t, t_channel], dim=1)

        return self.net(x)