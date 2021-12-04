import numpy as np

class Neural_Network:
    def __init__(self, input, hidden, output):
        self.weights_hidden = np.random.random((input, hidden))
        self.weights_output = np.random.random((hidden, output))
        self.bias_hidden = np.random.rand((1))
        self.bias_output = np.random.rand((1))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def predict(self, x):
        activate_hidden_layer = self.sigmoid(np.dot(x, self.weights_hidden) + self.bias_hidden)
        activate_output_layer = self.sigmoid(np.dot(activate_hidden_layer, self.weights_output) + self.bias_output)
        return activate_output_layer

    def error(self, error, curent_result, layer, bias):
        der = self.sigmoid_der(np.dot(curent_result, layer + bias))
        t_output_hidden = curent_result
        #delta of weights between hidden layer and output
        delta = np.dot(t_output_hidden.T, error * der)
        return delta, der
    
    def accuracy(self, test_vector_input, test_vector_output):
        count = 0     
        for i in range(len(test_vector_input)):
            activate_output_layer = self.predict(test_vector_input[i])
            output = np.zeros(len(test_vector_output[i]))
            max_value = np.argmax(activate_output_layer)
            output[max_value] = np.max(test_vector_output[i])
            if np.array_equal(test_vector_output[i], output):
                count += 1
        # Compute the average of correct instances
        accuracy = count/len(test_vector_input)  
        # Add it to the history vector (for plotting purposes)
        self.accuracy_history.append(accuracy*100)
        return accuracy*100


    def training(self, training_input, training_output, test_input, test_output,  lr, maxStep):
        self.accuracy_history = list()
        for epochs in range(maxStep):
            # calcul the intermadiate step
            layer1 = self.sigmoid(np.dot(training_input, self.weights_hidden) + self.bias_hidden)
            output = self.predict(training_input)

            # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
            d_weights2 = np.dot(layer1.T, (2*(training_output - output) * self.sigmoid_der(output)))
            d_weights1 = np.dot(training_input.T,  (np.dot(2*(training_output - output) * self.sigmoid_der(output), self.weights_output.T) * self.sigmoid_der(layer1)))
           
        
            # update the weights with the derivative (slope) of the loss function
            self.weights_hidden += d_weights1 * lr
            self.weights_output += d_weights2 * lr
            ac = self.accuracy(test_input, test_output) 
            #print(epochs, ac)
            if ac == 100 :break


    