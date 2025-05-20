import torch

# --- 1. Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device Setup ---")
print(f"Using device: {device}\n")

print('')
print('RESHAPE TENSOR ==================================')
original_tensor = torch.arange(12, dtype=torch.float32, device=device)
print(f"--- Original Tensor ---")
print(f"Tensor: {original_tensor}")

reshaped_tensor = original_tensor.reshape(3, 4)
print(f"--- Reshaping with .reshape(3, 4) ---")
print(f"Reshaped Tensor:\n{reshaped_tensor}")

print('')
print('TENSOR CONCATENATION ============================')
tensor_a = torch.tensor([[10, 20, 30],
                         [40, 50, 60]], dtype=torch.float32, device=device)
tensor_b = torch.tensor([[70, 80, 90],
                         [100, 110, 120]], dtype=torch.float32, device=device)

print(f"--- Concatenation (torch.cat()) ---")
print(f"Tensor A:\n{tensor_a}")
print(f"Tensor B:\n{tensor_b}")

concat_rows = torch.cat((tensor_a, tensor_b), dim=0)
print(f"Concatenated along dim=0 (rows):\n{concat_rows}")

concat_cols = torch.cat((tensor_a, tensor_b), dim=1)
print(f"Concatenated along dim=1 (columns):\n{concat_cols}")


# --- 5. Stacking Operation: torch.stack() ---
print('')
print('TENSOR STACKING =================================')
stack_tensor1 = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
stack_tensor2 = torch.tensor([4, 5, 6], dtype=torch.float32, device=device)
stack_tensor3 = torch.tensor([7, 8, 9], dtype=torch.float32, device=device)

print(f"--- Stacking (torch.stack()) ---")
print(f"Tensors to stack: {stack_tensor1}, {stack_tensor2}, {stack_tensor3}")

stacked_dim0 = torch.stack((stack_tensor1, stack_tensor2, stack_tensor3), dim=0)
print(f"Stacked along dim=0 (new 1st dim):\n{stacked_dim0}")

stacked_dim1 = torch.stack((stack_tensor1, stack_tensor2, stack_tensor3), dim=1)
print(f"Stacked along dim=1 (new 2nd dim):\n{stacked_dim1}")