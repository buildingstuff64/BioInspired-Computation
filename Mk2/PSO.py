import matplotlib.pyplot as plt
from tqdm import tqdm
from Mk2.ANN import ANN
import numpy as np
import random
import json
from multiprocessing import Manager, Queue


class PSO:
    def __init__(self, json_hyperparams, training_data, test_data):
        self.js = json_hyperparams
        self.swarm = [Particle(self.js) for _ in range(self.js['swarm_size'])]
        self.input_training_data, self.output_training_data = training_data
        self.input_test_data, self.output_test_data = test_data
        self.best_position = None
        self.debug = False

    def optimise(self, q=None):
        global_best_position = None
        epoch_best_fitness = []
        global_best_fitness = float('inf')

        for i in tqdm(range(self.js['iterations']), desc = "Training...", disable = not self.debug):
            if q is not None:
                q.put_nowait(i / self.js['iterations'])

            for particle in self.swarm:
                fitness = particle.ANN.calc_loss(self.input_training_data, self.output_training_data)
                particle.last_fitness = fitness

                if fitness < particle.best_fitness:
                    particle.personal_best = particle.position.copy()
                    particle.best_fitness = fitness

                if fitness < global_best_fitness:
                    global_best_position = particle.position.copy()
                    global_best_fitness = fitness


            informants_best_position = None
            informants_best_fitness = float('inf')
            for particle in self.swarm:
                informants = random.sample(self.swarm, self.js['informants'])
                for i in informants:
                    if i.best_fitness < informants_best_fitness:
                        informants_best_fitness = i.best_fitness
                        informants_best_position = i.position.copy()
                particle.update_vel(informants_best_position, global_best_position)
                particle.position = particle.position + particle.velocity
                particle.ANN.set_wb(particle.position)
            epoch_best_fitness.append(global_best_fitness)


        self.best_position = global_best_position
        return global_best_position, epoch_best_fitness

    def test(self):
        a = ANN(self.js['layer_sizes'], self.js['activation'], self.js['weight_range'], self.js['bias_range'])
        a.set_wb(self.best_position)
        LLL = []
        DDD = []
        for x, y in zip(a.forward_prop(self.input_test_data), self.output_test_data):
            LLL.append(a.huber_loss(y, x))
            print(f"{x.item():<5.4} : {y.item():<5.4}  loss : {LLL[-1].item():<5.4}  {(x == y).mean()}")
        print(f"Total Average Loss = {a.calc_loss(self.input_test_data, self.output_test_data)} \nTotal Divergence = {self.test_noprt()}")

    def test_noprt(self):
        a = ANN(self.js['layer_sizes'], self.js['activation'], self.js['weight_range'], self.js['bias_range'])
        a.set_wb(self.best_position)
        x = a.forward_prop(self.input_test_data)
        return np.abs(self.output_test_data - x).mean() ** 2



class Particle:
    def __init__(self, js):
        self.js = js
        self.ANN = ANN(js['layer_sizes'], js['activation'], js['weight_range'], js['bias_range'])
        self.position = self.ANN.get_wb()
        self.velocity = np.random.uniform(low=-1, high=1, size=self.position.size)
        #self.velocity.clip(min=-1, max=1)
        self.personal_best = np.copy(self.position)
        self.best_fitness = float('inf')
        self.last_fitness = float('inf')

    def update_vel(self, informants_best, global_best):
        r1 = np.random.uniform(low=0, high=self.js['beta'], size=self.velocity.size)
        r2 = np.random.uniform(low=0, high=self.js['gamma'], size=self.velocity.size)
        r3 = np.random.uniform(low=0, high=self.js['delta'], size=self.velocity.size)
        self.velocity = (
                (self.js['alpha'] * self.velocity) +
                r1 * (self.personal_best - self.position) +
                r2 * (informants_best - self.position) +
                r3 * (global_best - self.position)
        )
        #self.velocity.clip(min=-1, max=1)