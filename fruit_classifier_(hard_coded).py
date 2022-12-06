import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os

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

        self.biases = np.reshape(self.biases, (1,num_neurons))

    def forward(self,x):        
        return sigmoid(np.sum(self.weights * x) + self.biases)
        

class NeuralNetwork:
    def __init__(self,input_length):

    

        self.hidden_layer = Layer(3 ,input_length)
        self.output_layer = Layer(1,3)

        self.lr = .005
        
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
                x = np.tile(x,(layer.num_neurons, 1))
                output = layer.forward(x)
            else:
                # otherwise pass it the previous layer's output
                output = layer.forward(output_vectors[-1])

            output_vectors.append(output)
        
        return output_vectors

    def backpropagate(self, output, y):

        # output neuron update
        
        # this outputs a scalar
        output_delta = (output[2] - y) * output[2]* (1-output[2])
        # update the bias with this scalar
        self.output_layer.biases -= output_delta * self.lr

        # multiply the delta times the input to produce a 3-vector
        output_gradient = output_delta * output[1]
        # update the weights with this 3-vector
        self.output_layer.weights -= output_gradient * self.lr


        # hidden layer update

        # this outputs a 3-vector
        hidden_delta = output_gradient * self.output_layer.weights * (1 - output[1])
        # update the biases with this 3 vector
        self.hidden_layer.biases -= hidden_delta * self.lr
        
        # reshape it from (3,) to (3,1)
        hidden_delta = np.reshape(hidden_delta, (3,1))
        # reshape the input from (,10000) to (1,10000)
        output[0] = np.reshape(output[0], (1,10000))
        
        # this returns a (3,10000) matrix
        hidden_gradient = np.matmul(hidden_delta, output[0])
        # update the hidden layer weights with this matrix
        self.hidden_layer.weights -= hidden_gradient * self.lr

                   
image = np.asarray(Image.open('test.jpg').convert('L'))
input_vector = image.flatten()
input_length = len(input_vector)

nn = NeuralNetwork(input_length)




banana_path = r"C:\python\Computer Vision\Fruit_Classification\fruits-360_dataset\fruits-360\Training\Banana"
apple_path = r"C:\python\Computer Vision\Fruit_Classification\fruits-360_dataset\fruits-360\Training\Apple Braeburn"

bananas = os.listdir(banana_path)
apples = os.listdir(apple_path)


"""
np.asarray(Image.open('test.jpg').convert('L'))
input_vector = image.flatten()
image_class = 1
"""


def train():
    image_class = 0

    image_index = 0
    while image_index < len(bananas):

        if image_class == 0:
            image_class = 1
            image_path = apple_path + "\\"+apples[image_index]
            np.asarray(Image.open(image_path).convert('L'))
            input_vector = image.flatten()


        elif image_class == 1:
            image_class = 0
            image_path = banana_path + "\\"+bananas[image_index]
            np.asarray(Image.open(image_path).convert('L'))
            input_vector = image.flatten()
            
            image_index += 1
        
        output_vectors = nn.forward(input_vector)
        output_vectors.insert(0,input_vector)

                             
        prediction = output_vectors[-1]

        error = L2_loss(image_class, prediction)




        nn.backpropagate(output_vectors, image_class)

        prediction = nn.forward(input_vector)[-1]
        error = L2_loss(image_class, prediction)

        
        if image_index % 300 == 0:
            #print("IMAGE INDEX ", image_index)
            
            if image_class ==0:
                print(image_class, prediction)
                print(error)
    

epochs = 100
for epoch in range(epochs):
    train()




def display_image_from_vector(img):
    output_image = np.reshape(img, (input_shape))
    plt.imshow(output_image)
    plt.show()

