import torch
import torch.nn as nn

class UnsteadyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: x, y, t, amp, rot (5 dims)
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3) # u, v, p (No turbulence term for simplified unsteady)
        )

    def forward(self, x, y, t, amp, rot):
        inp = torch.cat([x, y, t, amp, rot], dim=1)
        return self.net(inp)
