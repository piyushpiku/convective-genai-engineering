import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# 1. MODEL DEFINITION
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: x(2) + t(1) = 3 dims
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x, t):
        t_vec = torch.ones(x.shape[0], 1).to(x.device) * t
        xt = torch.cat([x, t_vec], dim=1)
        return self.net(xt)

# 2. DATA UTILS
def get_source_data(n):
    data, _ = make_circles(n_samples=n, factor=0.5, noise=0.05)
    return torch.tensor(data, dtype=torch.float32)

def get_target_data(n):
    # Target: Square
    # Generate 4x the points to ensure we have enough after filtering
    data = np.random.uniform(-1.5, 1.5, size=(n * 4, 2)) 
    
    # Keep only those inside the [-1, 1] box
    mask = (np.abs(data[:, 0]) < 1) & (np.abs(data[:, 1]) < 1)
    
    valid_data = data[mask]
    
    # Safety check: if we still don't have enough, recurse (rare but safe)
    if len(valid_data) < n:
        return get_target_data(n)
        
    # Return exactly 'n' points
    return torch.tensor(valid_data[:n], dtype=torch.float32)

# 3. MAIN EXECUTION
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = VelocityNet().to(device)
    opt = optim.Adam(model.parameters(), lr=0.005)

    print("Training Shape Morphing (Circle -> Square)...")
    for i in range(2000):
        opt.zero_grad()
        x0 = get_source_data(1024).to(device)
        x1 = get_target_data(1024).to(device)
        t = torch.rand(1024, 1).to(device)
        
        # Flow Matching Loss
        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0
        v_pred = model(xt, t)
        
        loss = torch.mean((v_pred - ut)**2)
        loss.backward()
        opt.step()
        
        if i % 500 == 0:
            print(f"Step {i}: Loss {loss.item():.5f}")

    # Visualization
    print("Generating Flow...")
    with torch.no_grad():
        x_flow = get_source_data(1000).to(device)
        dt = 0.01
        for i in range(100):
            t_val = i * dt
            v = model(x_flow, t_val)
            x_flow += v * dt
            
    plt.scatter(x_flow.cpu()[:,0], x_flow.cpu()[:,1], alpha=0.5)
    plt.title("Morphed Output (Should be a Square)")
    plt.show()
