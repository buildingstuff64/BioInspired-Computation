from ANN import ANN
import numpy as np

def encode_decode_test():
    a = ANN([8, 16, 1], "sigmoid")
    a.forward_prop(np.random.randn(100, 8))
    weight_before = a.weights
    biaes_before = a.biases
    a.set_wb(a.get_wb())
    weight_after = a.weights
    biaes_after = a.biases
    for b, a in zip(weight_before, weight_after):
        assert np.array_equal(b, a)
        print(np.array_equal(b, a))
    for b, a in zip(biaes_before, biaes_after):
        assert np.array_equal(b, a)
        print(np.array_equal(b, a))


encode_decode_test()