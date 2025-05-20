# Lesson 03 | Tensor and Operation

>This lecture note introduces the fundamental unit of Deep Learning algorithms and how they interact with others through operations.
## Table of Contents
[1. Tensor](#1-tensor) \
&emsp;&emsp;[1.1. What is Tensor?](#11-what-is-tensor) \
&emsp;&emsp;[1.2. Tensor in PyTorch (Python)](#12-tensor-in-pytorch-pythons-library) \
[2. Operation](#2-operation)
## 1. Tensor
### 1.1. What is Tensor?

> In mathematics, a tensor is an algebraic object that describes a multilinear relationship between sets of algebraic objects related to a vector space. Tensors may map between different objects such as vectors, scalars, and even other tensors. [1]

There are many types of tensors, including scalars and vectors (which are the simplest tensors), dual vectors, multilinear maps between vector spaces, and even some operations such as the dot product.

<figure style="text-align: center;">
  <img src="img/tensor_example.png" alt="Tensor examples">
  <figcaption>Fig 01. Tensor examples: A visual representation of tensors.</figcaption>
</figure>

In brief, it can be simply understood that tensor is a generalization of scalars (0D), vectors (1D), matrices (2D), and higher-dimensional arrays (3D+).

### 1.2. Tensor in PyTorch
#### Tensor initialization
With the assistance of PyTorch, users can easily generate a tensor with the following syntax:
+ From `lists`/`arrays`: torch.tensor([...]), np.array([...])
+ Random: `torch.rand()`, `torch.randn()`, `torch.zeros()`, `torch.ones()`
+ Identity or custom initialization

For example:
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tensor_0D = torch.tensor(0)        # Create a 0D tensor
tensor_1D = torch.tensor([1])      # Create a 1D tensor
tensor_2D = torch.tensor([2, 3])   # Create a 2D tensor
tensor_2_2 = torch.tensor(         # Create a 2x2 tensor
    [[1, 0], [0, 1]]
) 
```
#### Tensor datatype
- `float32`, `float64` – for most model weights and inputs
- `int32`, `int64` – for labels, counters, indices
- `bool` – for masks, conditions

## 2. Operation
> In the context of tensors, an "operation" refers to any mathematical function or transformation that is applied to one or more tensors to produce new tensors. Think of it as how you manipulate numbers in basic arithmetic (addition, multiplication) but extended to these multi-dimensional data structures.
## REFERENCES
[1]: Wikipedia contributors. (2025, April 20). Tensor. Wikipedia. https://en.wikipedia.org/wiki/Tensor