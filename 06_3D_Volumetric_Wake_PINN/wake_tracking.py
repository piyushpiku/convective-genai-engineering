import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from flownet3d import FlowNet3D

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Model
    model = FlowNet3D().to(device)
    # Assume we load trained weights here
    # model.load_state_dict(torch.load("flownet3d.pth")) 
    
    # 2. Simulate "True" Object Location
    TRUE_LOC = torch.tensor([0.2, -0.1, 0.1]).to(device)
    print(f"Searching for object hidden at: {TRUE_LOC.tolist()}")
    
    # 3. Inverse Solver (Wake Tracking)
    # Guess starts at 0,0,0
    guess = torch.zeros(3, requires_grad=True, device=device)
    optimizer = optim.Adam([guess], lr=0.05)
    
    # Create dummy sensor data (In real life, this comes from sensors)
    # Here we just simulate the 'pull' towards the center
    
    print("Running 3D Wake Tracking...")
    for i in range(50):
        optimizer.zero_grad()
        
        # Loss: Distance between guess and truth (Simulated Wake Gradient)
        # In reality, this loss is ||Pred_Wake - Obs_Wake||^2
        # Here we simplify for the demo script:
        loss = torch.sum((guess - TRUE_LOC)**2)
        
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}: Current Guess {guess.detach().cpu().numpy()}")
            
    print(f"FINAL RESULT: Found object at {guess.detach().cpu().numpy()}")
