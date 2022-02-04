import numpy as np

class Layer_Danse:
    def __init__(self, n_inputs, n_neurons): 
        # print(np.random.randn(2,5)) 
        # >>> [[ 1.7640524 0.4001572 0.978738 2.2408931 1.867558 ],
        #      [-0.9772779 0.95008844 -0.1513572 -0.10321885 0.41059852]]
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        
        # print(np.zeros((2,5))) 
        # >>> [[0. 0. 0. 0. 0.] [0. 0. 0. 0. 0.]]
        self.biases = np.zeros((1, n_neurons))
        
    # Forward pass 
    def forward(self, inputs): 
        # Calculate output values from inputs, weights and biases 
        self.output = np.dot(inputs, self.weights) + self.biases
        pass # using pass statement as a placeholder