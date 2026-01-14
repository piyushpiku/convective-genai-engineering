import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TimeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 64))
        self.down1 = nn.Conv2d(1, 32, 3, padding=1)
        self.down2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        x1 = self.act(self.down1(x))
        x2 = self.act(self.down2(x1) + t_emb) # Inject time
        u1 = self.act(self.up1(x2))
        return self.out(u1 + x1) # Skip connection

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading MNIST Digits...")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = TimeUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training MNIST Flow...")
    for epoch in range(2): 
        for x1, _ in loader:
            x1 = x1.to(device)
            x0 = torch.randn_like(x1) # Source: Noise
            t = torch.rand(x1.shape[0], 1, 1, 1).to(device)
            
            xt = (1 - t) * x0 + t * x1
            ut = x1 - x0
            
            v_pred = model(xt, t.view(-1, 1))
            loss = torch.mean((v_pred - ut)**2)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")