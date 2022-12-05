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

    def forward(self,x):        
        return sigmoid(np.sum(self.weights * x) + self.biases)


class NeuralNetwork:
    def __init__(self,input_length):

    

        self.hidden_layer = Layer(5 ,input_length)
        self.output_layer = Layer(1,5)

        self.lr = .05
        
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
        for i in range(len(output)-1,0,-1):
            x = output[i-1]
            for neuron in range(len(self.layers[i-1].weights)):
                gradient = np.zeros(len(self.layers[i-1].weights[neuron]))
                for weight in range(len(self.layers[i-1].weights[neuron])):
                    gradient[weight] = -((y  -output[i][neuron]) * output[i][neuron] * (1-output[i][neuron]) * x[weight])
                    
                self.layers[i-1].weights[neuron] -= self.lr * gradient

                bias_gradient = -((y-output[i][neuron]) * output[i][neuron] * (1-output[i][neuron]))
                self.layers[i-1].biases[neuron] -= self.lr * bias_gradient
                         
                   
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
            print("IMAGE INDEX ", image_index)
            print(image_class, prediction)
            print(error)
    

epochs = 10
for epoch in range(epochs):
    train()




def display_image_from_vector(img):
    output_image = np.reshape(img, (input_shape))
    plt.imshow(output_image)
    plt.show()

