import torch

# Try different input shapes
x = torch.randn(4, 80, 6)  # Batch size of 4
print(f'Input shape: {x.shape}')

x_fixed = x.unsqueeze(1)
print(f'Shape after unsqueeze: {x_fixed.shape}')

print(f'Input mean: {x_fixed.mean()}')
print(f'Input std: {x_fixed.std()}')