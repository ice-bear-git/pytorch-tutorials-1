"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ || 
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
**Build Model** ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Build the Neural Network
===================

Neural networks comprise of layers/modules that perform operations on data. 
The `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ namespace provides all the building blocks you need to 
build your own neural network. Every module in PyTorch subclasses the `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. 
A neural network is a module itself that consists of other modules (layers). This nested structure allows for
building and managing complex architectures easily.

In the following sections, we'll build a neural network to classify images in the FashionMNIST dataset.

"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#############################################
# Get Device for Training
# -----------------------
# We want to be able to train our model on a hardware accelerator like the GPU, 
# if it is available. Let's check to see if 
# `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_ is available, else we 
# continue to use the CPU. 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

##############################################
# Define the Class !!!!!!!!!!!!!!!!!!!
# -------------------------
# We define our neural network by subclassing ``nn.Module``, and 
# initialize the neural network layers in ``__init__``. Every ``nn.Module`` subclass implements
# the operations on input data in the ``forward`` method. 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

##############################################
# We create an instance of ``NeuralNetwork``, and move it to the ``device``, and print 
# its structure.

model = NeuralNetwork().to(device)
print(model)


##############################################
# To use the model, we pass it the input data. This executes the model's ``forward``,
# along with some `background operations <https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866>`_. 
# Do not call ``model.forward()`` directly!
# 
# Calling the model on the input returns a 10-dimensional tensor with raw predicted values for each class.
# We get the prediction probabilities by passing it through an instance of the ``nn.Softmax`` module.

X = torch.rand(1, 28, 28, device=device)
logits = model(X) 
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
# Predicted class: tensor([1], device='cuda:0')

######################################################################
# --------------
#


##############################################
# Model Layers
# -------------------------
#
# Let's break down the layers in the FashionMNIST model. To illustrate it, we 
# will take a sample minibatch of 3 images of size 28x28 and see what happens to it as 
# we pass it through the network. 

input_image = torch.rand(3,28,28)
print(input_image.size())

# torch.Size([3, 28, 28])


##################################################
# nn.Flatten
# ^^^^^^^^^^^^^^^^^^^^^^
# We initialize the `nn.Flatten  <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_ 
# layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (
# the minibatch dimension (at dim=0) is maintained).
 
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#  nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784=28*28 pixel values 
# torch.Size([3, 784])

##############################################
# nn.Linear 
# ^^^^^^^^^^^^^^^^^^^^^^
# The `linear layer <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_
# is a module that applies a linear transformation on the input using its stored weights and biases.
#
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())



#################################################
# nn.ReLU
# ^^^^^^^^^^^^^^^^^^^^^^
# Non-linear activations are what create the complex mappings between the model's inputs and outputs.
# They are applied after linear transformations to introduce *nonlinearity*, helping neural networks
# learn a wide variety of phenomena.
#
# In this model, we use `nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_ between our
# linear layers, but there's other activations to introduce non-linearity in your model.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")


"""最常见的一种Activator Function， 可以把所有小于0的部分变成0，以此来破坏线性拟合的发生"""
# output
# Before ReLU: tensor([[-0.0355,  0.4598, -0.6033, -0.4020,  0.2564, -0.0131,  0.1266, -0.0853,
#          -0.2737, -0.2881, -0.2638, -0.1434,  0.4234, -0.2435,  0.1270,  0.0578,
#          -0.2097, -0.1305, -0.1978, -0.1190],
#         [ 0.1109,  0.2047, -0.5848, -0.2619,  0.1655, -0.1921, -0.3267, -0.3982,
#          -0.0389, -0.0965, -0.3208,  0.1393,  0.6055, -0.2739,  0.1808, -0.0544,
#          -0.4203, -0.3427,  0.1557, -0.0993],
#         [ 0.2247,  0.4551, -0.4619, -0.3765, -0.1110, -0.0804, -0.0426, -0.2185,
#           0.2792, -0.2306, -0.0773,  0.2841,  0.2210,  0.1206,  0.0541,  0.0560,
#          -0.5151, -0.2355,  0.1623,  0.1236]], grad_fn=<AddmmBackward>)


# After ReLU: tensor([[0.0000, 0.4598, 0.0000, 0.0000, 0.2564, 0.0000, 0.1266, 0.0000, 0.0000,
#          0.0000, 0.0000, 0.0000, 0.4234, 0.0000, 0.1270, 0.0578, 0.0000, 0.0000,
#          0.0000, 0.0000],
#         [0.1109, 0.2047, 0.0000, 0.0000, 0.1655, 0.0000, 0.0000, 0.0000, 0.0000,
#          0.0000, 0.0000, 0.1393, 0.6055, 0.0000, 0.1808, 0.0000, 0.0000, 0.0000,
#          0.1557, 0.0000],
#         [0.2247, 0.4551, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2792,
#          0.0000, 0.0000, 0.2841, 0.2210, 0.1206, 0.0541, 0.0560, 0.0000, 0.0000,
#          0.1623, 0.1236]], grad_fn=<ReluBackward0>)


#################################################
# nn.Sequential
# ^^^^^^^^^^^^^^^^^^^^^^
# `nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ is an ordered 
# container of modules. The data is passed through all the modules in the same order as defined. You can use
# sequential containers to put together a quick network like ``seq_modules``.
"""把之前定义好的每一层拼接起来"""
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

################################################################
# nn.Softmax
# ^^^^^^^^^^^^^^^^^^^^^^
# The last linear layer of the neural network returns `logits` - raw values in [-\infty, \infty] - which are passed to the
# `nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ module. The logits are scaled to values 
# [0, 1] representing the model's predicted probabilities for each class. ``dim`` parameter indicates the dimension along 
# which the values must sum to 1. 

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


#################################################
# Model Parameters
# -------------------------
# Many layers inside a neural network are *parameterized*, i.e. have associated weights 
# and biases that are optimized during training. Subclassing ``nn.Module`` automatically 
# tracks all fields defined inside your model object, and makes all parameters 
# accessible using your model's ``parameters()`` or ``named_parameters()`` methods. 
# 
# In this example, we iterate over each parameter, and print its size and a preview of its values. 
#


print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    
    
"""结果！通过model.named_parameters()与默认的model来窥探其内部的parameters"""
# Model structure:  NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#     (5): ReLU()
#   )
# ) 


# Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0242, -0.0002, -0.0232,  ..., -0.0203, -0.0267,  0.0066],
#         [-0.0322, -0.0067, -0.0182,  ..., -0.0134, -0.0055, -0.0030]],
#        device='cuda:0', grad_fn=<SliceBackward>) 

# Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([0.0346, 0.0123], device='cuda:0', grad_fn=<SliceBackward>) 

# Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0043, -0.0057,  0.0116,  ..., -0.0073, -0.0327, -0.0289],
#         [ 0.0311,  0.0312, -0.0264,  ...,  0.0413, -0.0338,  0.0071]],
#        device='cuda:0', grad_fn=<SliceBackward>) 

# Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0058,  0.0080], device='cuda:0', grad_fn=<SliceBackward>) 

# Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0140,  0.0323, -0.0378,  ..., -0.0163, -0.0349, -0.0080],
#         [ 0.0165,  0.0225, -0.0081,  ..., -0.0045, -0.0253, -0.0023]],
#        device='cuda:0', grad_fn=<SliceBackward>) 

# Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([0.0041, 0.0012], device='cuda:0', grad_fn=<SliceBackward>) 
    
    
    
######################################################################
# --------------
#

#################################################################
# Further Reading
# --------------
# - `torch.nn API <https://pytorch.org/docs/stable/nn.html>`_



