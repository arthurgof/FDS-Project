import numpy as np
from scipy.stats import norm



class Neural_Network:
    
    def __init__(self, input , hidden, output):
        layers = [input, hidden, output]
        self.weights = [np.random.uniform(-1, 1,(layers[i], layers[i+1]))for i in range(2)]
        self.bias = [np.random.uniform(-1, 1,(layers[i+1]))for i in range(2)]
        self.loss_history = list()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self,x):
        va = 2 * [0]
        ar = 2 * [0]
        va[0] = np.dot(x, self.weights[0]) + self.bias[0].T
        ar[0] = self.sigmoid(va[0])
        va[1] = np.dot(ar[0], self.weights[1]) + self.bias[1].T
        ar[1] = self.sigmoid(va[1])
        return ar[1], ar, va
      
    def error(self, x, y):
        a, ar, va = self.feedforward(x)
        return np.sum((y-a)**2)/x.shape[1]

    def derror(self, x, y):
        a, ar, va = self.feedforward(x)
        return (2*(a - y)/x.shape[1] * self.sigmoid_der(va[1])).T

    def backprog(self, x, y, ar, va): 
        wd = 2 * [0]
        bd = 2 * [0]
        delta = self.derror(x,y) # delta_L
        wd[1] = np.dot(delta, ar[0])
        bd[1] = np.sum(delta, axis = 1)
        delta = np.dot(self.weights[1],delta) * self.sigmoid_der(va[0]).T
        wd[0] = np.dot(delta, x)
        bd[0] = np.sum(delta,axis = 1) # reshape to keep dimensions and transpose
        return wd, bd
    
    def training(self, x, y, lr, itermax):
        for epoch in range(itermax):
            a, ar, va = self.feedforward(x)
            wd, bd = self.backprog(x,y, ar, va)
            for l in range(2):
                self.weights[l] -= lr * wd[l].T
                self.bias[l] -=  lr* bd[l].T
            err = self.error(x, y)
            self.loss_history.append(err)

    def accuracy(self, test_vector_input, test_vector_output):
        count = 0     
        for i in range(len(test_vector_input)):
            activate_output_layer, ar, va = self.feedforward(test_vector_input[i])
            output = np.zeros(len(test_vector_output[i]))
            max_value = np.argmax(activate_output_layer)
            output[max_value] = np.max(test_vector_output[i])
            if np.array_equal(test_vector_output[i], output):
                count += 1
        # Compute the average of correct instances
        accuracy = count/len(test_vector_input)  
        return accuracy*100

class NaiveBayes:        
    def __init__(self,train_INPUT, Train_output):
        self.mean = list()
        self.sdt = list()
        self.prior(np.array(Train_output))
        for i in range(10):
            lo = np.zeros(10)
            lo[i] = 1
            n = int(len(train_INPUT)/10)
            id = [i for i in range(n)]
            li = list()
            for y in id:
                l = train_INPUT[y+(n*i)]
                li.append(l)
            li = np.vstack(li)
            self.mean.append(np.array(li).mean(axis=0))
            self.sdt.append(np.array(li).std(axis=0))
            
        for i in range(10):
            self.sdt[i][self.sdt[i] == 0] = np.min(self.sdt[i][self.sdt[i] != 0.0])
              
    def prior(self, output):
        dic = {}
        count = np.unique(output, axis=0)
        for i in range(len(count)):
            dic[np.array2string(count[i])] = 0
        tot = 0
        for v in output:
            tot += 1
            dic[np.array2string(v)] += 1
        self.priorprob = np.zeros(len(count))
        for i in range(len(count)):
            self.priorprob[i] = dic[np.array2string(count[i])]/tot
                
    def predic(self, input):
        re = -1
        max = -1.22
        for i in range(10):
            sum = self.argmax(input,i)
            if np.isnan(sum):
                pass
            elif max == -1.22:
                re = i
                max = sum
            elif max < sum:
                re = i
                max = sum  
        return re
                
    def argmax(self,sample, id):
        posterior = np.sum(norm.logpdf(sample, loc = self.mean[id], scale = self.sdt[id])) + np.log(self.priorprob[id])
        return (posterior)
    
    def accuracy(self, test_vector_input, test_vector_output):
        count = 0     
        for i in range(len(test_vector_input)):
            y = self.predic(test_vector_input[i])
            output = np.zeros(len(test_vector_output[i]))
            output[y] = 1
            if np.array_equal(test_vector_output[i], output):
                count += 1
        # Compute the average of correct instances
        accuracy = count/len(test_vector_input)  
        return accuracy*100