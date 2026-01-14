import torch
import torch.optim as optim
from razor_model import RazorNet # Imports the class from file 1

def rans_loss_fn(model, x, y, amp, rot):
    x.requires_grad = True
    y.requires_grad = True
    
    out = model(x, y, amp, rot)
    u, v, p, nut = out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4]
    
    # Calculate Gradients (Autograd)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    
    # 1. Continuity Loss (Mass conservation)
    loss_mass = torch.mean((u_x + v_y)**2)
    
    # 2. Boundary Condition (Star Shape)
    # Define Star Distance Function
    theta = torch.atan2(y, x)
    r = torch.sqrt(x**2 + y**2)
    radius_limit = 0.2 + amp * torch.cos(5 * (theta - rot))
    mask_inside = torch.sigmoid(50 * (radius_limit - r))
    
    loss_bc = torch.mean(mask_inside * (u**2 + v**2))
    
    return loss_mass + 10.0 * loss_bc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RazorNet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training RANS PINN...")
    for i in range(1000):
        # Sample random points in domain
        x = torch.rand(1000, 1).to(device) * 2 - 1
        y = torch.rand(1000, 1).to(device) * 2 - 1
        amp = torch.rand(1000, 1).to(device) * 0.05 + 0.1
        rot = torch.rand(1000, 1).to(device) * 6.28
        
        loss = rans_loss_fn(model, x, y, amp, rot)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % 200 == 0:
            print(f"Step {i}, Loss: {loss.item():.5f}")
    
    print("Saving Model...")
    torch.save(model.state_dict(), "razor_rans.pth")
