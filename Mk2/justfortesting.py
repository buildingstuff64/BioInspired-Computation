import json

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

pso_data = json.load(open('data.json', 'r'))
pso = PSO(pso_data, (input_data, output_data), (input_data, output_data))
best_position, losses = pso.optimise()
print(best_position)
print(losses)
plt.plot(losses)
plt.show()