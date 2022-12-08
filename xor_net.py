import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os

import time

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivative_of_sigmoid(z):
    return np.exp(z)/((np.exp(z) + 1)**2)


def L2_loss(real, predicted):
    return (real - predicted)**2


class Layer:
    def __init__(self,num_neurons, input_length):
        self.num_neurons = num_neurons
        self.input_length = input_length
        
        # randomly initialize neuron weights

        self.weights = np.random.rand(num_neurons, input_length)
        self.biases = np.random.rand(num_neurons)

    def forward(self,x):
        return sigmoid(np.dot(self.weights, x) + self.biases)
        

class NeuralNetwork:
    def __init__(self,input_length):

    

        self.hidden_layer = Layer(3 ,input_length)
        self.output_layer = Layer(1,3)

        self.lr = 1
        
        self.layers = [
                self.hidden_layer,
                self.output_layer
            ]

    def forward(self,x):
        output_vectors = []
        
        for layer in self.layers:
            # if the first layer
            if not output_vectors:
                # pass it the input vector
                output = layer.forward(x)
            else:
                # otherwise pass it the previous layer's output
                output = layer.forward(output_vectors[-1])

            output_vectors.append(output)
        
        return output_vectors

    def backpropagate(self, output, y):

        E1 = output[2] - y
        dW2 = E1 * output[2] * (1-output[2])

        #print("E1: " + str(E1.shape))
        #print("dW2: " + str(dW2.shape))


        E2 = np.dot(np.reshape(dW2, (1,1)), self.output_layer.weights)
        dW1 = E2 * output[1] * (1-output[1])

        #print("E2: "+str(E2.shape))
        
        #print("dW1: "+str(dW1.shape))


        W2_update = np.dot(np.reshape(output[1], (3,1)), np.reshape(dW2, (1,1)))
        W1_update = np.dot(np.reshape(output[0], (2,1)), dW1)

        #print("W2_update: "+str(W2_update.shape))
        #print("W1_update: "+str(W1_update.shape))


        self.output_layer.biases = self.output_layer.biases - self.lr * dW2
        self.output_layer.weights = self.output_layer.weights - self.lr * W2_update.T
        #print("before "+str(self.hidden_layer.biases.shape))
        self.hidden_layer.biases = self.output_layer.biases - self.lr * np.reshape(dW1,(3,))
        self.hidden_layer.weights = self.hidden_layer.weights - self.lr * W1_update.T
        #print("after "+str(self.hidden_layer.biases.shape))
        
        """
        # this outputs a scalar
        output_delta = (output[2] - y) * output[2]* (1-output[2])
        # multiply the delta times the input to produce a 3-vector

        


        output_gradient = -np.matmul(
            np.reshape(output_delta, (1,1)),
            np.reshape(output[1], (1,3)))


        # this outputs a 3-vector
        hidden_delta =  np.matmul(output_delta, self.output_layer.weights) * output[1] * (1-output[1])


        # this returns a (3,2) matrix
        hidden_gradient = np.matmul(
            np.reshape(hidden_delta, (3,1)),
            np.reshape(output[0], (1,2)))

        
        
        # update the output bias
        self.output_layer.biases += -output_delta * self.lr
        # update the output weights
        self.output_layer.weights += -output_gradient * self.lr

        # update the hidden layer biases
        self.hidden_layer.biases += -hidden_delta * self.lr
        # update the hidden layer weights
        self.hidden_layer.weights += -hidden_gradient * self.lr
        """


input_length = 2

nn = NeuralNetwork(input_length)


data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]



def train(epoch):

    image_pair = ["",""]
    
    image_class = 0

    image_index = 0
    for input_vector, class_label in data:


        input_vector = np.array(input_vector)
        output_vectors = nn.forward(input_vector)
        output_vectors.insert(0,input_vector)
                             
        prediction = output_vectors[-1]

        error = L2_loss(image_class, prediction)

        nn.backpropagate(output_vectors, class_label)

        prediction = nn.forward(input_vector)[-1]

        error = L2_loss(image_class, prediction)

        if epoch % 20000 == 0:
            print(str(input_vector) +" : " + str(class_label) + " : "+ str(prediction[0]))

epochs = 200000
for epoch in range(epochs):
    #print("\n")
    train(epoch)


