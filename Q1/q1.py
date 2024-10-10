import numpy as np


np.random.seed(0)
class ffnn:

    class layer:
        def __init__(self, a, b):
            self.weights = np.random.randn(b, a) * 0.01
            self.bias = np.zeros((b, 1))

        def __str__(self):
            return f"{self.weights} \n {self.bias}"


    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.hidden_layer_count = len(layer_sizes) - 2

    def init_layers(self):
        self.layers = []
        for i in range(1, len(self.layer_sizes)):
            self.layers.append(self.layer(self.layer_sizes[i - 1], self.layer_sizes[i]))

        print(len(self.layers))
        for i in self.layers:
            print(i)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propigation(self, inputs):
        x = inputs
        for i in self.layers:
            x = self.sigmoid(np.dot(i.weights, x) + i.bias)
        return x

nerualnet = ffnn([4, 5, 2])
nerualnet.init_layers()
print()
print(nerualnet.forward_propigation(np.array([4, 6, 2, 6])))