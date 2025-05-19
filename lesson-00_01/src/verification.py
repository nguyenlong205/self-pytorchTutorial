import torch
print("PyTorch version")
print(torch.__version__, end = '\n\n')

print("My first PyTorch programme")
x = torch.rand(5, 3)
print(x)
print("completed")

print("Check of CUDA version and its availability")
print(torch.version.cuda)
print(torch.cuda.is_available(), end = '\n\n')