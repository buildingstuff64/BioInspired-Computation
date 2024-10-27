import json
from random import random

import pandas as pd
import numpy as np

from Mk2.ANN import ANN
from DataHandler import *
from PSO import PSO
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def Normilisedata(data):
    _d = []
    for d in data:
        min_val, max_val = (0, 1)
        d = np.array(d, dtype = float)
        data_max = np.max(d, axis = 0)

        # Avoid division by zero in case of constant input values
        denom = data_max - np.min(d, axis = 0)
        denom = np.where(denom == 0, 1, denom)  # Replace 0s with 1 to avoid division by zero

        normalized_data = (d - np.min(d, axis = 0)) / denom  # Scale to [0, 1]
        normalized_data = normalized_data * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
        _d.append(normalized_data)
    return _d

file = pd.read_csv('concrete_data.csv')
#data, data_min, data_max =  normalize(file)

#file, data_min, data_max = normalize(file)
input_data = np.array(file.iloc[:, : 8].to_numpy())
output_data = np.array(file.iloc[:, 8: ].to_numpy())
input_data, data_min, data_max  = normalize(input_data)

# a = ANN([8, 16, 1], "sigmoid")
# b = a.forward_prop(input_data)
# print(b)
# c = np.mean((b - output_data)**2)
# print(c)
#
# print(f"descaled data \n {unnormalize(output_data, data_min, data_max)}  \n {unnormalize(b, data_min, data_max)}")

losses = pd.DataFrame(
    {
        "alpha": [],
        "beta": [],
        "gamma": [],
        "delta": [],
        "loss": []
    }
)
pso_data = json.load(open('data.json', 'r'))
for i in range(1):
    pso = PSO(pso_data, (input_data, output_data), (input_data, output_data))
    best_pos, loss = pso.optimise()
    losses = losses._append(
        {
            "alpha": pso_data['alpha'],
            "beta": pso_data['beta'],
            "gamma": pso_data['gamma'],
            "delta": pso_data['delta'],
            "loss": loss[-1]
        },
        ignore_index=True
    )
    print(f"iteration {i} done : loss = {loss[-1]}  \n a:{pso_data['alpha']} b:{pso_data['beta']} g:{pso_data['gamma']} d:{pso_data['delta']}")
    a = ANN(pso_data['layer_sizes'], "sigmoid")
    a.set_wb(best_pos)
    out = a.forward_prop(input_data)
    print(f"expected = {output_data[:3]} : output = {out[:3]}")
    #print(f"expected = {unnormalize(output_data[:3], data_min, data_max)} : output = {unnormalize(out[:3], data_min, data_max)}")
    plt.plot(loss)
    plt.show()

# print(min(losses['loss']))
# losses.to_csv("output.csv")

# a = ANN([8, 16, 1], "sigmoid")
# a.set_wb(best_pos)
# out = a.forward_prop(input_data)
# print(f"expected = {output_data[0]} : output = {out[0]}")
# print(f"expected = {unnormalize(output_data[0], data_min, data_max)} : output = {unnormalize(out[0], data_min, data_max)}")
# print(loss[-1])
#
# plt.plot(loss)
# plt.show()

# a = ANN([8, 16, 1], "sigmoid")
# data = Normilisedata([[540.0 ,0.0 ,0.0 ,162.0 ,2.5 ,1040.0 ,676.0 ,28]])
# a.test_forward_pass(data)


# pso_data = json.load(open('data.json', 'r'))
# pso = PSO(pso_data, (input_data, output_data), (input_data, output_data))
# best_pos, loss = pso.optimise()
# a = ANN([8, 100, 1], "sigmoid")
# a.set_wb(best_pos)
# out = a.forward_prop(input_data)
# for f, o in zip(out, output_data):
#     print(f"expected = {unnormalize(o, data_min, data_max)} : output = {unnormalize(f, data_min, data_max)}")
#     print(f"expected = {o} : output = {f}\n")
#
# print(f"accuracy {1 - loss[-1]}%")


#plt.plot(loss)
#plt.show()

print("Finished")

