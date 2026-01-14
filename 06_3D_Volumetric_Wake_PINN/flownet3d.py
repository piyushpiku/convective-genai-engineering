import torch
import torch.nn as nn

class FlowNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: x, y, z (3 dims)
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4) # u, v, w, p
        )

    def forward(self, x, y, z):
        inp = torch.cat([x, y, z], dim=1)
        return self.net(inp)
