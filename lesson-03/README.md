# Lesson 03 | Tensor and Operation

>This lecture note introduces the fundamental unit of Deep Learning algorithms and how they interact with others through operations.
## Table of Contents
[1. Tensor](#1-tensor) \
&emsp;&emsp;[1.1. What is Tensor?](#11-what-is-tensor) \
&emsp;&emsp;[1.2. Tensor in PyTorch (Python)](#12-tensor-in-pytorch-pythons-library) \
[2. Operation](#2-operation) \
&emsp;&emsp;[2.1. Mathematical Operation](#21-mathematical-operation) \
&emsp;&emsp;[2.2. Agregate Function](#22-aggregate-function) \
&emsp;&emsp;[2.2. Agregate Function](#22-aggregate-function) 


## 1. Tensor
### 1.1. What is Tensor?

> In mathematics, a tensor is an algebraic object that describes a multilinear relationship between sets of algebraic objects related to a vector space. Tensors may map between different objects such as vectors, scalars, and even other tensors. [1]

There are many types of tensors, including scalars and vectors (which are the simplest tensors), dual vectors, multilinear maps between vector spaces, and even some operations such as the dot product.

![alt](img/tensor_example.png)

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

### 2.1. Mathematical operation
They encompass:
- **Numerical operations**
    + Addition `+`
    + subtraction `-`
    + Multiplication `*` or `torch.mul()`
    + Division `/`
- **Comparison Operations**
    + Equality (`==` or `torch.eq()`)
    + Greater than (`>` or `torch.gt()`)
    + Less than (`<` or `torch.lt()`)
- **Mathematical functions** (exponential `torch.exp()`, logarithm `torch.log()`, sin `sin()`, absolute value `torch.abs()`, ...)
- **Matrix Operation**
    + Transpose `.T` or `torch.transpose()`
    + Matrix Multiplication `@` or `torch.mamul()`
    + Dot product `torch.dot()`

### 2.2. Aggregate function
They include **sum** `torch.sum()`, **max** `torch.max()` / **min** `torch.min()`, **mean** `torch.mean()`, ...

For example:
```python
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
```
Now it should Return:
```text
Sum: 10.0
Min: 1.0
Max: 4.0
Mean: 2.5
```

### 2.3. Shape adjustment
- **Reshaping Operations**: `torch.reshape()` or `torch.view()`
- **Concatentation**: `torch.cat((a, b), dim = 0)` for concatinate along rows, or `torch.cat((a, b), dim = 1)` for concatinate along columns.
- **Stacking**: `torch.stack()`

> *What are the differences between **concatenation** and **stack**?*

Concatenation
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.cat((a, b), dim=0) # Concatenate along rows
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])
d = torch.cat((a, b), dim=1) # Concatenate along columns
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])
```
Stacking
```python
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
c = torch.stack((a, b), dim=0) # Stacks them as rows
# tensor([[1, 2],
#         [3, 4]])
d = torch.stack((a, b), dim=1) # Stacks them as columns
# tensor([[1, 3],
#         [2, 4]])
```
## REFERENCES
[1] Wikipedia contributors. (2025, April 20). Tensor. Wikipedia. https://en.wikipedia.org/wiki/Tensor \
[2] Học viện spacedev. (n.d.). https://spacedev.vn/resources/docs/pytorch/tensor-operations