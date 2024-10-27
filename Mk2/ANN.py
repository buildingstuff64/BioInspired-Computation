import numpy as np
from scipy.special import expit

class ANN:
    def __init__(self, layer_sizes, activation_function):
        self.layers = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(self.layers)-1):
            self.weights.append(np.random.rand(self.layers[i], self.layers[i+1]))
            self.biases.append(np.random.rand(1, self.layers[i+1]))

    def set_wb(self, wb):
        w_len = sum([w.size for w in self.weights])
        b_len = sum([b.size for b in self.biases])
        self.set_weights(wb[:w_len])
        self.set_biases(wb[-b_len:])

    def get_wb(self):
        return np.concatenate([self.get_weights_flat(), self.get_biases_flat()]).flatten()

    def set_weights(self, flat_weights):
        self.weights = ANN.fltn(flat_weights, self.weights)

    def get_weights_flat(self):
        return np.concatenate([w.flatten() for w in self.weights])

    def set_biases(self, flat_biases):
        self.biases = ANN.fltn(flat_biases, self.biases)

    def get_biases_flat(self):
        return np.concatenate([b.flatten() for b in self.biases])

    def fltn(x, y):
        new_w = []
        for w in y:
            a, b = w.shape
            new_w.append(x[:a * b].reshape(a, b))
            x = x[a * b:]
        return new_w

    def forward_prop(self, inputs):
        for i in range(len(self.layers)-1):
            wxb = np.dot(inputs, self.weights[i]) + self.biases[i]
            act = self.sigmoid(wxb)
            inputs = act if i == len(self.layers) - 1 else wxb
        return inputs

    def sigmoid(self, x):
        return 1 / (1 + expit(-x))

    def calc_loss(self, inputs, expected_outputs):
        fwd = self.forward_prop(inputs)
        return np.abs(expected_outputs - fwd).mean()

    def test_forward_pass(self, test_in):
        print(f"test data in = {test_in}")
        for i in range(len(self.layers)-1):
            wxb = np.dot(test_in, self.weights[i]) + self.biases[i]
            print(f"inputs {i} * weights + biases = \n{wxb}")
            act = self.sigmoid(wxb)
            test_in = act if i == len(self.layers)-1 else wxb
            print(f"activation {i} = {test_in}")
        return test_in



