import json
import math
import random

import numpy as np
from alive_progress import alive_bar

from NerualNet import *


class PSO:
    def __init__(self, json_config, data):
        self.js = json.load(json_config)
        self.training_data = (self.Normilisedata(data[0][0]), data[0][1])
        self.test_data = (self.Normilisedata(data[1][0]), data[1][1])
        self.particles = [Particle(self.js, self.training_data) for x in range(self.js['swarm_size'])]
        self.inertia = self.js['inertia_weight']
        self.c1 = self.js['cognitive_constant']
        self.c2 = self.js['social_constant']

    def Optimise(self):

        best_fitness_in_itr = []
        best_particle_in_itr = None
        with alive_bar(self.js['iterations'] * 2 * self.js['swarm_size'], force_tty=True) as bar:
            for i in range(self.js['iterations']):
                best_fitness = math.inf
                for particle in self.particles:
                    f = particle.fitness()
                    if f < best_fitness:
                        best_fitness = f
                        best_particle_in_itr = particle
                        particle.best_position = particle.position
                    bar()

                best_fitness_in_itr.append(best_fitness)
                for particle in self.particles:
                    r1 = np.random.rand(*particle.position.shape) * self.js['randomness']
                    r2 = np.random.rand(*particle.position.shape) * self.js['randomness']
                    pt = []
                    [pt.append(x.position) for x in np.random.choice(np.array(self.particles), 1)]
                    rnd = np.random.choice(np.array(self.particles), 1)
                    cog_vel = self.c1 * r1 * (particle.best_position - particle.position)
                    soc_vel = self.c2 * r2 * (pt[0] - particle.position)
                    particle.velocity = (self.inertia * particle.velocity) + cog_vel + soc_vel

                    particle.position += particle.velocity
                    particle.set_position(particle.position)
                    bar()
                bar()



        return best_particle_in_itr, best_fitness_in_itr

    def Test(self, p):
        for o, e in zip(self.test_data[0], self.test_data[1]):
            out = p.forward_prop(np.array(o, dtype=float))
            print(f"expected = {e} : output = {out}")
        print(f"expected = anything : output is {p.forward_prop(np.random.rand(8))}")

    def Normilisedata(self, data):
        _d = []
        for d in data:
            print(d)
            min_val, max_val = (0, 1)
            d = np.array(d, dtype=float)
            data_max = np.max(d, axis = 0)

            # Avoid division by zero in case of constant input values
            denom = data_max - np.min(d, axis = 0)
            denom = np.where(denom == 0, 1, denom)  # Replace 0s with 1 to avoid division by zero

            normalized_data = (d - np.min(d, axis = 0)) / denom  # Scale to [0, 1]
            normalized_data = normalized_data * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
            _d.append(normalized_data)
        return _d



class Particle:
    def __init__(self, js, training_data):
        self.velocity = random.random()
        self.nn = NeuralNet(js['layers'], js['activation'])
        self.best_fitness = math.inf
        self.position = self.get_position()
        self.best_position = np.copy(self.position)
        self.training_data = training_data

    def fitness(self):
        b = self.nn.calc_loss(np.array(self.training_data[0], dtype=float), np.array(self.training_data[1], dtype=float))
        if b < self.best_fitness:
            self.best_fitness = b
            self.best_position = self.get_position()
        return b

    def forward_prop(self, input):
        return self.nn.fwd_prop(input)

    def get_position(self):
        return np.concatenate(self.nn.get_all_params())

    def set_position(self, pos):
        self.nn.set_all_params(pos)
