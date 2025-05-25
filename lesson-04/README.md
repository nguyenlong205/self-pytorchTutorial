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





## REFERENCES
[1] *What is a Neural Network & How Does It Work?* (May $25^{th}$ 2025). Google Cloud. https://cloud.google.com/discover/what-is-a-neural-network \
[2] *Neural Networks and How They Work in Natural Language Processing* (May $25^{th}$ 2025). Pangeanic. https://blog.pangeanic.com/neural-networks-and-how-they-work-in-natural-language-processing \
[3] *Module*, PyTorch, https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
