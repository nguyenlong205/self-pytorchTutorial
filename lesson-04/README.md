# Lesson 04 | Neural Network with PyTorch

> This lecture provides a foundational understanding of building and training neural networks using PyTorch. We'll cover essential components like `nn.Module`, linear layers, activation and loss functions, and delve into the critical processes of optimization and backpropagation, with a brief introduction to creating custom layers.



## Table of Contents
[1. Introduction to Neural Networks with Pytorch](#1-introduction-to-neural-networks-with-pytorch) \
&emsp;&emsp;[1.1. What is Neural Network?](#11-what-is-neural-network) \
&emsp;&emsp;[1.2. How Neural Network works?](#12-how-neural-network-works) \
&emsp;&emsp;[1.3. Why Neural Network?](#13-why-neural-network) \
&emsp;&emsp;[1.4. Uses of Neural Network in Natural Language Processing (NLP)](#14-uses-of-neural-network-in-natural-language-processing-nlp) \
[2. Building blocks of Neural Networks with PyTorch](#2-building-blocks-of-neural-networks-with-pytorch) \
&emsp;&emsp;[2.1. `nn.Module`](#21-nnmodule) \
&emsp;&emsp;[2.2. Fully connected linear transformation layer](#22-fully-connected-linear-transformation-layer) \
&emsp;&emsp;&emsp;&emsp;[2.2.1 Mathematical foundation](#221-mathematical-foundation) \
&emsp;&emsp;&emsp;&emsp;[2.2.2. PyTorch approach](#222-pytorch-approach)\
&emsp;&emsp;&emsp;&emsp;[2.2.3. Attribute of Fully Connected Linear Transformation](#223-attribute-of-fully-connected-linear-transformation)

[REFERENCES](#references) 


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

Learning in neural network, actually is the process of establishment and adjustment of weights of the connections between nodes in a neural system. Those processes are commonly called as ***training***. These weights will be optimised while training occurs. Particularly, this process is to refine its weights to minimize errors between its predictions and the actual values.

An ***activation function*** adds non-linearity to a neural network layer’s output, enabling the model to learn complex patterns. Without it, the network would act like a simple linear model. Common activation functions include `ReLU` (Rectified Linear Unit), which outputs zero for negative inputs and the input itself if positive, `sigmoid`, which squashes values between 0 and 1, and `tanh`, which outputs values between -1 and 1. Activation functions enable neural networks to approximate complex functions and solve a wide variety of tasks such as classification, regression, and more.

A technique called ***backpropagation*** *(will be introduced later in this lecture)* facilitates this weight adjustment. Backpropagation calculates the error in the output and propagates it back through the network, iteratively adjusting weights until a desired accuracy level is achieved. Once trained, the network can make predictions on new data, such as identifying cats in images.

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
import torch
import torch.nn as nn
import torch.nn.functional as F
class MySimpleModel(nn.Module):
    def __init__(self):
        super().__init__()  # Call nn.Module's constructor to properly initialize the module
        # Define layers
        self.lin1 = nn.Linear(4, 2)
        self.lin2 = nn.Linear(2, 1)
    def forward(self, x):
        # Connect nodes of the layers
        x = self.lin1(x)
        x = self.lin2(x)
        return x
```
The code snippet above illustrates the following neural network architecture.
![alt text](/lesson-04/img/fully_connected_model_diagram.png)

More sophisticated models will be introduced and analysed in detail in the following lessons.

### 2.2. Fully connected linear transformation layer
#### 2.2.1. Mathematical foundation
`nn.Linear` in the context of deep learning typically refers to a linear transformation or fully connected layer in a neural network. It's a fundamental building block in many neural network architectures. It applies an affine linear transformation to the incoming data: 
$$
y=xA^T+b
$$
where:
- $x$ is the input tensor.
- $A$ is the weight matrix (often denoted as ***weight*** in implementations).
- $b$ is the bias vector (often denoted as bias).
- $y$ is the output tensor.

For example, let

- $x = \begin{bmatrix} x_1 & x_2 & x_3 & x_4 \end{bmatrix}$
- $A = \begin{bmatrix} A_{11} & A_{12} & A_{13} & A_{14} \\ A_{21} & A_{22} & A_{23} & A_{24} \end{bmatrix}$
- $b = \begin{bmatrix} b_1 & b_2 \end{bmatrix}$

then the transformed vector $y = xA^T+b$ has a size of $1 \times 2$.

#### 2.2.2. PyTorch approach
In PyTorch, `nn.Linear(in_features, out_features)` represents a fully connected linear transformation (also called a dense layer). Mathematically, it implements the function:

Its parameters consist of:
- `in_features` (`int`) – size of each input sample
- `out_features` (`int`) – size of each output sample
- `bias` (`bool`) – If set to `False`, the layer will not learn an additive bias. Default: `True`

Here is an example:
```py
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
```
It should return the following result.
```
Input tensor shape: torch.Size([3, 10])
Output tensor shape: torch.Size([3, 5])

Weight matrix shape: torch.Size([5, 10])
Bias vector shape: torch.Size([5])
```
#### 2.2.3. Attribute of Fully Connected Linear Transformation
## REFERENCES
[1] *What is a Neural Network & How Does It Work?* (May $25^{th}$ 2025). Google Cloud. https://cloud.google.com/discover/what-is-a-neural-network \
[2] *Neural Networks and How They Work in Natural Language Processing* (May $25^{th}$ 2025). Pangeanic. https://blog.pangeanic.com/neural-networks-and-how-they-work-in-natural-language-processing \
[3] *Module*, PyTorch, https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html \
[4] *Linear*, PyTorch, https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html \
[5] *What is Fully Connected Layer in Deep Learning?*, GeekForGeeks, https://www.geeksforgeeks.org/what-is-fully-connected-layer-in-deep-learning/