# Lesson 04 | Neural Network with PyTorch

> This lecture provides a foundational understanding of building and training neural networks using PyTorch. We'll cover essential components like `nn.Module`, linear layers, activation and loss functions, and delve into the critical processes of optimization and backpropagation, with a brief introduction to creating custom layers.



## Table of Contents
[1. Introduction to Neural Networks with Pytorch](#1-introduction-to-neural-networks-with-pytorch) \
&emsp;&emsp;[1.1. What is Neural Network?](#11-what-is-neural-network) \
&emsp;&emsp;[1.2. How Neural Network works?](#12-how-neural-network-works) \
&emsp;&emsp;[1.3. Why Neural Network?](#13-why-neural-network) \
&emsp;&emsp;[1.4. Uses of Neural Network in Natural Language Processing (NLP)](#14-uses-of-neural-network-in-natural-language-processing-nlp)




## 1. Introduction to Neural Networks with Pytorch
### 1.1. What is Neural Network?
> *Neural networks are a type of machine learning algorithm inspired by the structure and function of the human brain. They consist of interconnected nodes or neurons that process data, much like how neurons in the brain communicate. These networks can learn to recognize patterns and make predictions based on the data they are trained on.*

![alt text](img/neural_network_example.png)

**Neurons (Nodes)**: These are the fundamental processing units. Each neuron receives inputs, performs a simple computation, and then passes an output to other neurons.

**Layers**: Neurons are typically arranged in layers:
- Input Layer: This is where the raw data (e.g., pixel values of an image, words in a sentence) enters the network.
- Hidden Layers: These are the "thinking" layers in between the input and output. They perform complex transformations and learn abstract features from the data. Deep neural networks have many hidden layers.
- Output Layer: This layer produces the final result of the network, such as a classification (e.g., "cat" or "dog") or a prediction (e.g., a stock price).

### 1.2. How Neural Network works?

Learning in neural network, actually is the process of establishment and adjustment of weights of the connections between nodes in a neural system. Those processes are commonly called as *training*. These weights will be optimised while training occurs. Particularly, this process is to refine its weights to minimize errors between its predictions and the actual values.

A technique called backpropagation *(will be introduced later in this lecture)* facilitates this weight adjustment. Backpropagation calculates the error in the output and propagates it back through the network, iteratively adjusting weights until a desired accuracy level is achieved. Once trained, the network can make predictions on new data, such as identifying cats in images.

### 1.3. Why Neural Network?
Neural networks are becoming an essential tool for many businesses and organizations. Here are some reasons why they are so important:

- **Automation**: Automating tasks that were previously done by humans, such as customer service, data analysis, and image processing can be assisted by neural networks. This can save businesses time and money.
- **Improved Decision-Making**: Businesses can make better decisions by providing insights that would be difficult or impossible to obtain using traditional methods.
- **Increased Efficiency**: Business processes can be improved by automating tasks, reducing errors, and improving decision-making.
- **New Products and Services**: Businesses are enabled to create new products and services that would not be possible without AI.

### 1.4. Uses of Neural Network in Natural Language Processing (NLP)

Neural networks have given NLP models a huge capacity for understanding and simulating human language. They have allowed machines to predict words and address topics that were not part of the learning process. To achieve this performance in NLP processes, the neural networks must be trained with large amounts of documents (corpora) according to the type of text or language to be processed.

In NLP language models, neural networks act in the early stages, transforming vocabulary words into vectors. They act based on the principle that, in a text, the meaning of a certain word is associated with the words found around it. These vectors are used in simple operations to provide reasonable results at the semantic level. 

Neural networks are also employed in natural language technology to enable computers to successfully perform the NLP process. In this way, texts or documents can be processed, information extracted and the meaning of the data determined. 
- For example, chatbots or sentiment analysis for social media comments.


## 2. Building Blocks of Neural Networks with PyTorch
To manage and function neural networks, Pytorch provides `torch.nn` as a sub-module to perform these tasks. In these lectures, if something likes `nn.<some_thing>`, it is similar to `torch.nn.<something>`. The following sub-modules will be also imported as:
```py
import torch.nn as nn
```
### 2.1. `nn.Module`
`nn.Module` is the fundamental base class for all neural network modules in PyTorch. `nn.Module` is the base class for all neural network modules in PyTorch. Any neural network component, from a single layer to a complete deep learning model, should inherit from `nn.Module`.

Here is an example of defining a Neural Network Model
```py
Okay, here's a complete, simple example of a neural network using nn.Module in PyTorch, covering its definition, instantiation, a forward pass, and how parameters are managed. We'll use a very basic network for a regression task (predicting a continuous value).

Python

import torch
import torch.nn as nn
import torch.nn.functional as F # Often used for activation functions like F.relu

# --- 1. Define the Neural Network Class using nn.Module ---
class SimpleRegressor(nn.Module):
    """
    A simple neural network for regression tasks.
    It takes one input feature and predicts one output value.
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Always call the constructor of the parent class (nn.Module)
        super(SimpleRegressor, self).__init__()

        # Define the layers of the network (these are the 'bricks')
        # nn.Linear: Performs a linear transformation (input @ Weight.T + Bias)
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # We can also define activation functions if they are part of the model structure
        # In this case, we'll use F.relu in the forward pass, but sometimes you might
        # instantiate them like self.activation = nn.ReLU() if they have internal state
        # or if you prefer to assign them as a module attribute.

    def forward(self, x):
        """
        Defines the forward pass of the network.
        This describes how data flows through the defined layers.
        """
        # Input 'x' passes through the hidden layer
        x = self.hidden_layer(x)
        # Apply an activation function (Rectified Linear Unit)
        x = F.relu(x) # We use F.relu here, from torch.nn.functional
        # The result passes through the output layer
        x = self.output_layer(x)
        return x

# --- 2. Instantiate the Model ---
# Let's define the sizes for our network
input_dim = 1   # We have 1 input feature (e.g., age, temperature)
hidden_dim = 10 # Our hidden layer will have 10 neurons
output_dim = 1  # We want to predict 1 output value (e.g., price, quantity)

# Create an instance of our SimpleRegressor model
model = SimpleRegressor(input_dim, hidden_dim, output_dim)

print("--- Model Architecture ---")
print(model) # Printing the model shows its structure based on __init__

# --- 3. Simulate Input Data ---
# Let's create some dummy input data.
# A typical PyTorch tensor for a model usually has a batch dimension first.
# torch.randn(batch_size, num_features)
batch_size = 5
dummy_input = torch.randn(batch_size, input_dim) # 5 samples, each with 1 feature

print(f"\n--- Dummy Input Data ---")
print(f"Shape: {dummy_input.shape}")
print(f"Data:\n{dummy_input}")

# --- 4. Perform a Forward Pass ---
# To make a prediction, simply call the model instance with your input data.
# This implicitly calls the `forward` method you defined.
output = model(dummy_input)

print(f"\n--- Output After Forward Pass ---")
print(f"Shape: {output.shape}")
print(f"Data:\n{output}") # These are the raw predictions from the network

# --- 5. Inspect Model Parameters ---
# nn.Module automatically tracks all learnable parameters (weights and biases).
print("\n--- Model Parameters ---")
for name, param in model.named_parameters():
    print(f"Parameter Name: {name}")
    print(f"  Shape: {param.shape}")
    print(f"  Requires Gradient: {param.requires_grad}") # True means it will be updated during training
    # print(f"  Value (first 5 elements if >5): {param.data.flatten()[:5]}") # uncomment to see actual values

# --- 6. Move Model to a Device (e.g., GPU if available) ---
# This demonstrates nn.Module's device management.
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n--- Moving model and data to {device} ---")
    model.to(device)
    dummy_input = dummy_input.to(device)
    output_on_gpu = model(dummy_input)
    print(f"Output shape on GPU: {output_on_gpu.shape}")
    print(f"Output device: {output_on_gpu.device}")
else:
    print("\n--- CUDA (GPU) not available. Model remains on CPU. ---")

print("\n--- Example Complete ---")
```





## REFERENCES
[1] *What is a Neural Network & How Does It Work?* (May $25^{th}$ 2025). Google Cloud. https://cloud.google.com/discover/what-is-a-neural-network \
[2] *Neural Networks and How They Work in Natural Language Processing* (May $25^{th}$ 2025). Pangeanic. https://blog.pangeanic.com/neural-networks-and-how-they-work-in-natural-language-processing \
[3] *Module*, PyTorch, https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
