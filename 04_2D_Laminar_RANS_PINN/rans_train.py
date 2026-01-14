import torch
import torch.nn as nn
import numpy as np

class RazorNet(nn.Module):
    def __init__(self, sigma=30.0):
        super().__init__()
        # Fourier Features matrix (buffer to save it)
        self.register_buffer('B', torch.randn(4, 64) * sigma) 
        
        self.net = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 4) # u, v, p, nut (turbulence)
        )

    def forward(self, x, y, amp, rot):
        # Concatenate inputs
        inp = torch.cat([x, y, amp, rot], dim=1)
        # Fourier Projection
        proj = 2 * np.pi * inp @ self.B
        inp_f = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(inp_f)
