import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],3) 
        self.weights2   = np.random.rand(3,8)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_der(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_der(self.output), self.weights2.T) * self.sigmoid_der(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def training(self, maxStep):
        for i in range(maxStep):
            self.feedforward()
            self.backprop()