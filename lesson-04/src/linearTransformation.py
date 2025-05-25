import torch
import torch.nn as nn

# Define a linear layer
# It expects an input with 10 features and will output 5 features
linear_layer = nn.Linear(in_features=10, out_features=5)

# Create a dummy input tensor (batch size of 3, 10 features)
# Imagine 3 samples, each with 10 numerical attributes
input_tensor = torch.randn(3, 10)
print("Input tensor shape:", input_tensor.shape)

# Pass the input through the linear layer
output_tensor = linear_layer(input_tensor)
print("Output tensor shape:", output_tensor.shape)

# Accessing the learned parameters
print("\nWeight matrix shape:", linear_layer.weight.shape)
print("Bias vector shape:", linear_layer.bias.shape)