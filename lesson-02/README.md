# Introduction to PyTorch

This lecture introduces the detailed description of PyTorch, lists the tasks/jobs that require PyTorch, explain the reasons for using PyTorch, compares between Deep Learning libraries (TensorFlow and PyTorch), and introduce a life-cycle of a deep learning project.

## Table of Contents
[1. What is PyTorch?](#1-what-is-pytorch) \
[2. General principle of PyTorch](#2-general-principle-of-pytorch) \
&emsp;[2.1. Tensor: Basic unit of Deep Learning](#21-tensor-basic-unit-of-deep-learning) \
&emsp;[2.2. Autograd](#22-autograd) \
&emsp;[2.3. Modules and Models](#23-modules--models)


## 1. What is PyTorch
PyTorch is an open-source machine learning library developed by Facebook's AI Research lab (FAIR). It is widely used for **deep learning tasks** such as computer vision, natural language processing (NLP), and reinforcement learning. PyTorch offers dynamic computation graphs and a Pythonic interface, making it intuitive for researchers and developers alike. A number of pieces of deep learning software are built on top of PyTorch, including Tesla Autopilot, Uber's Pyro, Hugging Face's Transformers, and Catalyst.

PyTorch provides two high-level features:
- Tensor computing (like NumPy) with strong acceleration via graphics processing units (GPU)
- Deep neural networks built on a tape-based automatic differentiation system



## 2. General principle of PyTorch
The general principle of PyTorch is to provide a flexible, dynamic framework for building and training machine learning models - especially *neural networks—using automatic differentiation* and *tensor operations*. It’s particularly well-suited for research and deep learning development.

> What is *Neural Networks using automatic differentiation*? 

*Neural networks, specifically multi-layer perceptrons, are used for function approximation in machine learning. The basic idea is to define a large nonlinear function f(x, θ), parameterized by a vector θ, and then minimize a chosen loss function L measuring how well f predicts the desired output ŷ for given inputs x̂. This minimization is typically done using local optimization algorithms on the empirical risk, which is the sum of the loss over the data. A regularization penalty can also be added to θ.*

*Because the network consists of many layers and nonlinearities, direct application of calculus rules to compute derivatives becomes impractical. This is where automatic differentiation (autodiff), specifically reverse-mode autodiff, becomes essential.*

> *Tensors and Tensor Operations?*

*The central unit of data in PyTorch (and also other DL frameworks) is ***Tensor*** – a set of values shaped into an array of one or more dimensions. tf.Tensor are very similar to multidimensional arrays. We will show it more details in the upcoming lesson (Lesson 03)*

*While tensors allow you to store data, operations (ops) allow you to manipulate that data. PyTorch (`torch`) provides a wide variety of ops suitable for linear algebra and machine learning that can be performed on tensors.*

### 2.1. Tensor: Basic unit of Deep Learning

Generally, **tensors** are multi-dimentional arrays and similar to `numpy`'s arrays. The difference between them is the ability of running on GPUs. Here is an example of a **tensor**.
```python
import torch
# Create a 2x3 tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)
```
It should return the following result
```
tensor([[1, 2, 3],
        [4, 5, 6]])
```


### 2.2. Autograd
PyTorch's autograd provides automatic differentiation for all operations on tensors. Set requires_grad=True to track computations. For example:
```py
x = torch.tensor([2.], requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # Gradient of y w.r.t x
```
In the example above, `y.backward()` computes the derivative of `y` with respect to `x`, which is
$$
\frac{dy}{dx}
$$
using reverse-mode automatic differentiation. So `x.grad` will be:
$$
\left.\frac{dy}{dx}\right|_{x=2} = 2x = 2 \times 2 = 4
$$
As a result, it returns:
```
tensor([4.])
```

## 2.3. Modules & Models
In PyTorch, models are Python classes that inherit from torch.nn.Module. This base class provides:
- A way to store layers (like nn.Linear, nn.Conv2d, etc.)
- Automatic parameter tracking
- A method to define forward computation

In PyTorch, nn.Module provides powerful capabilities for building and managing models. One key advantage is parameter management: by defining layers inside an nn.Module, all parameters (such as weights and biases) are automatically registered and accessible via model.parameters(). This makes it straightforward to pass them to an optimizer for training.

Another benefit is support for nested models. You can include other modules as attributes within your custom model, enabling the construction of complex, deep architectures in a clean and modular way.

Lastly, nn.Module makes serialization easy. You can save the model’s learned parameters using torch.save(model.state_dict()), and later load them back with model.load_state_dict(...), allowing for convenient checkpointing and deployment of models.

```bash
git add .
git commit -m "update lesson 02"
git push
```

## References

[1] Wikipedia contributors. (2025, April 19). PyTorch. Wikipedia. https://en.wikipedia.org/wiki/PyTorch \
[2] Domke, J. (n.d.). Automatic Differentiation and Neural Networks. https://people.cs.umass.edu/~domke/courses/sml2010/07autodiff_nnets.pdf \
[3] GeeksforGeeks. (2021, August 20). Tensors and operations. GeeksforGeeks. https://www.geeksforgeeks.org/tensors-and-operations/ \