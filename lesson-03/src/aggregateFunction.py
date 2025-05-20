import torch

# Create a 2D tensor (a matrix)
data = torch.tensor([[1., 2.],
                     [3., 4.]])

# Calculate aggregate values
Sum = torch.sum(data)
Max = torch.max(data)
Min = torch.min(data)
Mean = torch.mean(data)

# Print results
print(f'Sum: {Sum}')
print(f'Min: {Min}')
print(f'Max: {Max}')
print(f'Mean: {Mean}')