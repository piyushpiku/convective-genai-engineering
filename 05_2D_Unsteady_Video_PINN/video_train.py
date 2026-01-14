import torch
import torch.optim as optim
from unsteady_model import UnsteadyNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnsteadyNet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training Unsteady Video Model...")
    for i in range(500):
        # Sample x, y, t
        x = torch.rand(1000, 1).to(device) * 2 - 1
        y = torch.rand(1000, 1).to(device) * 2 - 1
        t = torch.rand(1000, 1).to(device) # Time 0 to 1
        amp = torch.ones(1000, 1).to(device) * 0.1
        rot = torch.zeros(1000, 1).to(device)
        
        # Forward
        out = model(x, y, t, amp, rot)
        u, v = out[:,0:1], out[:,1:2]
        
        # Simple loss: Minimize velocity at boundary (moving circle example)
        r = torch.sqrt(x**2 + y**2)
        # Assume moving object center: x_c = 0.5 * sin(2*pi*t)
        xc = 0.5 * torch.sin(6.28 * t)
        dist = torch.sqrt((x - xc)**2 + y**2)
        mask = torch.sigmoid(50 * (0.2 - dist))
        
        loss = torch.mean(mask * (u**2 + v**2))
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss.item()}")
