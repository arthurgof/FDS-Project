import numpy as np

class Neural_Network:
    def __init__(self, input, hidden, output):
        self.weights_hidden = np.random.random((input, hidden))
        self.weights_output = np.random.random((hidden, output))
        self.bias_hidden = np.random.rand(1)
        self.bias_output = np.random.rand(1)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def predict(self, y):
        activate_hidden_layer = self.sigmoid(np.dot(y, self.weights_hidden)+ self.bias_hidden)
        activate_output_layer = self.sigmoid(np.dot(activate_hidden_layer, self.weights_output)+ self.bias_output)
        return activate_output_layer

    def error(self, error, curent_result, layer, bias):
        der = self.sigmoid_der(np.dot(curent_result, layer) + bias)
        t_output_hidden = curent_result
        #delta of weights between hidden layer and output
        delta = np.dot(t_output_hidden.T, error * der)
        return delta, der
    
    def accuracy(self, test_vector_input, test_vector_output):
        count = 0     
        for i in range(len(test_vector_input)):
            activate_output_layer = self.predict(test_vector_input[i])
            output = np.zeros(8)
            max_value = np.argmax(activate_output_layer)
            output[max_value] = 1
            if np.array_equal(test_vector_output[i], output):
                count += 1
        # Compute the average of correct instances
        accuracy = count/len(test_vector_input)  
        # Add it to the history vector (for plotting purposes)
        self.accuracy_history.append(accuracy*100)


    def training(self, training_input, training_output, lr, maxStep):
        self.accuracy_history = list()
        for epochs in range(maxStep):
            for iteration in range(epochs):
                # predict of hd layer with current network
                output_hidden = self.sigmoid(np.dot(training_input, self.weights_hidden) + self.bias_hidden)
                output_output_layer = self.sigmoid(np.dot(output_hidden, self.weights_output) + self.bias_output)

                #Calculat the error from the backtrack
                error_ot = output_output_layer - training_output
                delta_ho, der_hidden_output = self.error(error_ot, 
                                                        output_hidden, 
                                                        self.weights_output, 
                                                        self.bias_output)
                #Update weights of hidden layer -> find the weight matrix for the hidden␣
                error_io = error_ot * der_hidden_output
                error_ho = np.dot(error_io, self.weights_output.T)

                delta_ih, der_input_hidden = self.error(error_ho, 
                                                        training_input, 
                                                        self.weights_hidden, 
                                                        self.bias_hidden)
                #update weights
                self.weights_output -= lr * delta_ho
                self.weights_hidden -= lr * delta_ih
            self.accuracy(training_input, training_output)


    