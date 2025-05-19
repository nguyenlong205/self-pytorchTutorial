import torch

# ===================================================
# Create a 2x3 tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)

# ===================================================
x = torch.tensor([2.], requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # Gradient of y w.r.t x