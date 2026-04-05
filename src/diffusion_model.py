import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self .net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, y_t, t):
        x = x.unsqueeze(1)
        y_t = y_t.unsqueeze(1)
        t = t.unsqueeze(1)

        input = torch.cat([x, y_t, t], dim=1)
        print(input.shape)

        return self.net(input)