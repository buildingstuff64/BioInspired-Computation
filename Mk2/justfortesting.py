import json
import random

import matplotlib.pyplot as plt

from ANN import ANN
from PSO import PSO
import numpy as np
import pandas as pd

file = pd.read_csv('concrete_data.csv')
#data, data_min, data_max =  normalize(file)

#file, data_min, data_max = normalize(file)
input_data = np.array(file.iloc[:, : 8].to_numpy())
output_data = np.array(file.iloc[:, 8: ].to_numpy())

input_data_train = input_data[int(len(input_data)*0.7):]
output_data_train = output_data[int(len(output_data)*0.7):]

input_data_test = input_data[-int(len(input_data)*0.3):]
output_data_test = output_data[-int(len(output_data)*0.3):]

pso_data = json.load(open('data.json', 'r'))

# for i in range(50):
#     pso_data['beta'] = random.random()
#     pso_data['gamma'] = random.random()
#     pso_data['delta'] = random.random()
#
#     pso = PSO(pso_data, (input_data_train, output_data_train), (input_data_test, output_data_test))
#     best_position, losses = pso.optimise()
#     loss = pso.test_noprt()
#     print(f"{pso_data['beta']:<4.4} : {pso_data['gamma']:<4.4} : {pso_data['delta']:<4.4} \n ->{loss:<4.4} \n")


pso = PSO(pso_data, (input_data_train, output_data_train), (input_data_test, output_data_test))
pso.debug = True
best_position, losses = pso.optimise()
pso.test()
figure, axis = plt.subplots(2)

# For Sine Function
axis[0].plot(losses[10:])
axis[0].set_title("total losses")

# For Cosine Function
axis[1].plot(losses[-10:])
axis[1].set_title("")

plt.show()


