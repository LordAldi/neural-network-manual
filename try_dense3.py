import nnfs
import numpy as np
from Activation.common import Activation_Softmax_Loss_CategoricalCrossentropy
from Layer.dense_layer import Layer_Dense
from Activation.relu import Activation_ReLU
from nnfs.datasets import spiral_data

from Optimizer.optimizer_adam import Optimizer_Adam

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer 
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)


# Train in loop 
for epoch in range(10001): 
    # Perform a forward pass of our training data through this layer 
    dense1.forward(X) 
    
    # Perform a forward pass through activation function 
    # takes the output of first dense layer here 
    activation1.forward(dense1.output) 
    # Perform a forward pass through second Dense layer 
    # # takes outputs of activation function of first layer as inputs 
    dense2.forward(activation1.output) 
    # Perform a forward pass through the activation/loss function 
    # # takes the output of second dense layer here and returns loss 
    loss = loss_activation.forward(dense2.output, y) 
    # Calculate accuracy from output of activation2 and targets 
    # # calculate values along first axis 
    predictions = np.argmax(loss_activation.output, axis=1) 
    if len(y.shape) == 2: 
        y = np.argmax(y, axis=1) 
    accuracy = np.mean(predictions==y) 
    
    if not epoch % 100: 
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate}')
        # Backward pass 
    loss_activation.backward(loss_activation.output, y) 
    dense2.backward(loss_activation.dinputs) 
    activation1.backward(dense2.dinputs) 
    dense1.backward(activation1.dinputs) 
    # Update weights and biases 
    optimizer.pre_update_params() 
    optimizer.update_params(dense1) 
    optimizer.update_params(dense2) 
    optimizer.post_update_params()