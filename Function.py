import numpy as np

class Neural_Network:
    def __init__(self, layers:list):
        self.n = len(layers) - 1
        self.weights = [np.random.uniform(-1, 1,(layers[i], layers[i+1]))for i in range(self.n)]
        self.bias = [np.random.uniform(-1, 1,(layers[i+1]))for i in range(self.n)]
        self.va = self.n * [0]
        self.a = self.n * [0]
        self.accuracy_history = list()


    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self,x):
        self.va[0] = np.dot(x, self.weights[0]) + self.bias[0].T
        self.a[0] = self.sigmoid(self.va[0])
        for i in range(self.n - 1):
            self.va[i + 1] = np.dot(self.a[i], self.weights[i + 1]) + self.bias[i + 1].T
            self.a[i + 1] = self.sigmoid(self.va[i + 1])
        return self.a[self.n - 1]
    
    def error(self, x, y):
        a = self.feedforward(x)
        return np.sum((y-a)**2)/x.shape[1]

    def derror(self, x, y):
        a = self.feedforward(x)
        return (2*(a - y)/x.shape[1] * self.sigmoid_der(self.va[self.n-1])).T

    def backprog(self, x, y): 
        wd = self.n * [0]
        bd = self.n * [0]
        delta = self.derror(x,y) # delta_L
        for i in range(self.n-1, 0,-1):   #starts at second last layer and counts backwards
            wd[i] = np.dot(delta, self.a[i-1])
            bd[i] = np.sum(delta, axis = 1)
            delta = np.dot(self.weights[i],delta) * self.sigmoid_der(self.va[i-1]).T
        wd[0] = np.dot(delta, x)
        bd[0] = np.sum(delta,axis = 1) # reshape to keep dimensions and transpose
        return wd, bd
    
    def training(self, x, y, lr, itermax):
        for epoch in range(itermax):
            a = self.feedforward(x)
            wd, bd = self.backprog(x,y)
            for l in range(self.n):
                self.weights[l] -= lr * wd[l].T
                self.bias[l] -=  lr* bd[l].T
            err = self.error(x, y)
            print(epoch, err)
            self.accuracy_history.append(err)

    def accuracy(self, test_vector_input, test_vector_output):
        count = 0     
        for i in range(len(test_vector_input)):
            activate_output_layer = self.feedforward(test_vector_input[i])
            output = np.zeros(len(test_vector_output[i]))
            max_value = np.argmax(activate_output_layer)
            output[max_value] = np.max(test_vector_output[i])
            if np.array_equal(test_vector_output[i], output):
                count += 1
        # Compute the average of correct instances
        accuracy = count/len(test_vector_input)  
        return accuracy*100

if __name__ == "__main__":
    # ==> https://towardsdatascience.com/backpropagation-in-neural-networks-6561e1268da8

    NN = Neural_Network([8, 8, 8])
    input = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]])

    output = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]]).T

    NN.training(input, output, 0.1, 100)