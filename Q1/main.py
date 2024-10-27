import numpy as np
import matplotlib.pyplot as plt
from ParticleSwarmOptimization import PSO
from TrainingDataHandler import TrainingDataHandler
from ParticleSwarmOptimization import NeuralNet

train_data, test_data = TrainingDataHandler('concrete_data.csv').get()

# n = NeuralNet([8, 16, 1], "sigmoid")
# input = np.array([[540.0 ,0.0 ,0.0 ,162.0 ,2.5 ,1040.0 ,676.0 ,28], [540.0 ,0.0 ,0.0 ,162.0 ,2.5 ,1055.0 ,676.0 ,28]])
# ouput = n.fwd_prop(input)
# loss = n.calc_loss(input, [[79.99], [61.89]])
# print(f"input = {input} \n output = {ouput} \n loss = {loss}")

pso = PSO(open('data.json', 'r'), [train_data, test_data])
out , loss = pso.Optimise()
pso.Test(out)
print(f"output = {out.best_fitness} \n loss = {loss}")
plt.plot(loss)
plt.show()
