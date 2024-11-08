import numpy as np
from scipy.special import expit

class ANN:
    def __init__(self, layer_sizes, activation_function, wr, br):
        self.layers = layer_sizes
        self.weights = []
        self.biases = []
        self.activation_function = activation_function

        for i in range(len(self.layers)-1):
            self.weights.append(np.random.uniform(wr[0], wr[1], (self.layers[i], self.layers[i+1])))
            self.biases.append(np.random.uniform(br[0], br[1],(1, self.layers[i+1])))

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
            act = eval(f"self.{self.activation_function}(wxb)")
            inputs = act if i == len(self.layers) - 1 else wxb
        return inputs

    def sigmoid(self, x):
        return 1 / (1 + expit(-x))

    def ReLU(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def calc_loss(self, inputs, expected_outputs):
        fwd = self.forward_prop(inputs)
        # return np.abs(expected_outputs - fwd).mean() ** 2
        return self.huber_loss(expected_outputs, fwd)

    def huber_loss(self, y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta

        # For small errors, use squared loss
        squared_loss = 0.5 * error ** 2

        # For large errors, use linear loss
        linear_loss = delta * (np.abs(error) - 0.5 * delta)

        # Combine both
        loss = np.where(is_small_error, squared_loss, linear_loss)

        return np.mean(loss)

    def test_forward_pass(self, test_in):
        print(f"test data in = {test_in}")
        for i in range(len(self.layers)-1):
            wxb = np.dot(test_in, self.weights[i]) + self.biases[i]
            print(f"inputs {i} * weights + biases = \n{wxb}")
            act = self.sigmoid(wxb)
            test_in = act if i == len(self.layers)-1 else wxb
            print(f"activation {i} = {test_in}")
        return test_in



