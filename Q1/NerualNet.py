import numpy as np
from manim import sigmoid


class NeuralNet:
    def __init__(self, layer_sizes, activation):
        self.num_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.activation = activation

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def fwd_prop(self, inputs):
        x = inputs
        for i, l in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = l.calc_layer(x, "")
            else:
                x = l.calc_layer(x, self.activation)
        return x

    def calc_loss(self, inputs, expected):
        fwd = self.fwd_prop(inputs)
        print(f"shape match {fwd.shape} == {expected.shape}")
        return np.mean((expected - fwd)**2)

    def get_all_params(self):       
        return [self.get_weights(), self.get_biases()]

    def set_all_params(self, x):
        for l in self.layers:
            a, b = l.info
            l.weights = x[:a*b].reshape(a, b)
            x = x[a*b:]
        for l in self.layers:
            a, b = l.info
            l.biases = x[:b].reshape(1, b)
            x = x[b:]

    def get_weights(self):
        return np.concatenate([l.weights.flatten() for l in self.layers])

    def get_biases(self):
        return np.concatenate([l.biases.flatten() for l in self.layers])

    def __str__(self):
        str = ""
        for l in self.layers:
            str += l.__str__()
        return str

class Layer:
    def __init__(self, a, b):
        self.weights = np.random.rand(a, b)
        self.biases = np.zeros((1, b))
        self.info = (a, b)

    def calc_layer(self, x, activation):
        # print(f"input = {x} \n Weights = {self.weights}")
        # n = np.dot(x, self.weights)
        # print(f"n = {n} \n biases = {self.biases}")
        # p = n + self.biases
        # print(f"n + biaes = {p}")
        # act = self.getActivation(p, activation)
        # print(f"activation = {act}")
        return self.getActivation(np.dot(x, self.weights) + self.biases, activation)

    def getActivation(self, x, name):
        if name == "sigmoid":
            return self.sigmoid(x)
        elif name == "ReLU":
            return self.relu(x)
        elif name == "tanh":
             return self.hyperbolic_tangent(x)
        else:
            return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def hyperbolic_tangent(self, x):
        return np.tanh(x)

    def __str__(self):
        return f"Layer Size = {self.info} \nWeights\n{self.weights}\nBias's\n{self.biases}\n"
