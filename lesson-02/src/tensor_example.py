import torch

# ===================================================
# Create a 2x3 tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)

# ===================================================
# Define two tensors
a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([3.], requires_grad=True)

# Compute result
c = a * b
c.backward()

# Gradients
print(a.grad)  # Gradient w.r.t a