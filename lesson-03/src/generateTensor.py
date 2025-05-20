import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tensor_0D = torch.tensor(0)        # Create a 0D tensor
tensor_1D = torch.tensor([1])      # Create a 1D tensor
tensor_2D = torch.tensor([2, 3])   # Create a 2D tensor
tensor_2_2 = torch.tensor(         # Create a 2x2 tensor
    [[1, 0], [0, 1]]
) 
print(tensor_0D)
print(tensor_1D)
print(tensor_2D)
print(tensor_2_2)