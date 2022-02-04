import nnfs
import numpy as np
from dense_layer import Layer_Danse
from nnfs.datasets import spiral_data
nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Danse(2, 3)

dense1.forward(X)
print(dense1.output[:5])