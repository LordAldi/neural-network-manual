import nnfs
import numpy as np
from Activation.softmax import Activation_Softmax
from Layer.dense_layer import Layer_Dense
from Activation.relu import Activation_ReLU
from nnfs.datasets import spiral_data

from Loss.cross_entropy import Loss_CategoricalCrossentropy
nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)
print('loss:', loss)