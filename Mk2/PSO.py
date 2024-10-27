from tqdm import tqdm

from ANN import ANN
import numpy as np
import random
import json

class PSO:
    def __init__(self, json_hyperparams, training_data, test_data):
        self.js = json_hyperparams
        self.swarm = [Particle(self.js) for _ in range(self.js['swarm_size'])]
        self.input_training_data, self.output_training_data = training_data
        self.input_test_data, self.output_test_data = test_data

    def optimise(self):
        global_best_position = None
        epoch_best_fitness = []

        for i in tqdm(range(self.js['iterations']), desc = "Training..."):
            global_best_fitness = float('inf')
            for particle in self.swarm:
                fitness = particle.ANN.calc_loss(self.input_training_data, self.output_training_data)
                particle.last_fitness = fitness

                if fitness < particle.best_fitness:
                    particle.personal_best = particle.position
                    particle.best_fitness = fitness

                if fitness < global_best_fitness:
                    global_best_position = particle.position
                    global_best_fitness = fitness


            informants_best_position = None
            informants_best_fitness = float('inf')
            for particle in self.swarm:
                informants = random.sample(self.swarm, self.js['informants'])
                for i in informants:
                    if i.best_fitness < informants_best_fitness:
                        informants_best_fitness = i.best_fitness
                        informants_best_position = i.position
                particle.update_vel(informants_best_position, global_best_position)
                particle.position = particle.position + particle.velocity
                particle.ANN.set_wb(particle.position)
            epoch_best_fitness.append(global_best_fitness)


        return global_best_position, epoch_best_fitness




class Particle:
    def __init__(self, js):
        self.js = js
        self.ANN = ANN(js['layer_sizes'], js['activation'])
        self.position = self.ANN.get_wb()
        self.velocity = np.random.rand(*self.position.shape) * js['random']
        self.personal_best = np.copy(self.position)
        self.best_fitness = float('inf')
        self.last_fitness = float('inf')

    def update_vel(self, informants_best, global_best):
        r1 = np.random.rand(*self.velocity.shape) * self.js['random']
        r2 = np.random.rand(*self.velocity.shape) * self.js['random']
        r3 = np.random.rand(*self.velocity.shape) * self.js['random']
        self.velocity = (
                self.js['alpha'] * self.velocity +
                self.js['beta'] * r1 * (self.personal_best - self.position) +
                self.js['gamma'] * r2 * (informants_best - self.position) +
                self.js['delta'] * r3 * (global_best - self.position)
        )