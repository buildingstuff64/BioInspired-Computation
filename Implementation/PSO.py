from tqdm import tqdm
from Implementation.ANN import ANN
import numpy as np
import random
import pandas as pd


class PSO:
    def __init__(self, json_hyperparams, training_data, test_data):
        """
        Initilises the particle swarm with correct hyperparameters and training & test data
        :param json_hyperparams: hyperparameters for pso (json format)
        :param training_data: training data used to train the swarm
        :param test_data: test data only used for testing
        """
        self.js = json_hyperparams
        self.swarm = [Particle(self.js) for _ in range(self.js['swarm_size'])]
        self.input_training_data, self.output_training_data = training_data
        self.input_test_data, self.output_test_data = test_data
        self.best_position = None
        self.debug = False

    def optimise(self, q=None):
        """
        Runs the Particle Swarm Optimisation algorithm using the hyperparameters
        :param q: is used for progress bar updates in the GUI
        :return:
            list: list of all weights and bias's for the best position the pso found (flattened)
            list: a list of losses across all iterations
        """
        global_best_position = None
        epoch_best_fitness = []
        global_best_fitness = float('inf')

        for i in tqdm(range(self.js['iterations']), desc = "Training...", disable = not self.debug):
            if q is not None:
                q.put_nowait(i / self.js['iterations'])

            #finds the best global fitness and the particles personal best fitness
            for particle in self.swarm:
                fitness = particle.ANN.calc_loss(self.input_training_data, self.output_training_data)

                if fitness < particle.best_fitness:
                    particle.personal_best = particle.position.copy()
                    particle.best_fitness = fitness

                if fitness < global_best_fitness:
                    global_best_position = particle.position.copy()
                    global_best_fitness = fitness



            informants_best_position = None
            informants_best_fitness = float('inf')
            for particle in self.swarm:
                #find the best fitness in the particles informants
                informants = random.sample(self.swarm, self.js['informants'])
                for i in informants:
                    if i.best_fitness < informants_best_fitness:
                        informants_best_fitness = i.best_fitness
                        informants_best_position = i.position.copy()

                #updates the particles velocity & position aswell as updating the ANN
                particle.update_vel(informants_best_position, global_best_position)
                particle.position = particle.position + particle.velocity
                particle.ANN.set_wb(particle.position)
            epoch_best_fitness.append(global_best_fitness)


        self.best_position = global_best_position
        return global_best_position, epoch_best_fitness

    def test(self, print_values=True):
        """
        Is used for testing the pso created
        :param print_values: set false if you don't want results printed to the console
        :return:
            float: average loss
            dataframe: dataframe containing the results
        """
        a = ANN(self.js['layer_sizes'], self.js['activation'], self.js['weight_range'], self.js['bias_range'])
        a.set_wb(self.best_position)
        LLL = []
        fwd = a.forward_prop(self.input_test_data)
        d = pd.DataFrame({'input':[], 'output':[], 'loss':[]})
        for x, y in zip(fwd, self.output_test_data):
            LLL.append(a.huber_loss(y, x))
            if print_values:
                print(f"{x.item():<5.4} : {y.item():<5.4}  loss : {LLL[-1].item():<5.4}")
            d.loc[len(d)] = [x.item(), y.item(), LLL[-1].item()]
        avg_loss = a.calc_loss(self.input_test_data, self.output_test_data)
        if print_values:
            print(f"Total Average Loss = {avg_loss}")
        return avg_loss, d

class Particle:
    def __init__(self, js):
        self.js = js
        self.ANN = ANN(js['layer_sizes'], js['activation'], js['weight_range'], js['bias_range'])
        self.position = self.ANN.get_wb()
        self.velocity = np.random.uniform(low=-1, high=1, size=self.position.size)
        #self.velocity.clip(min=-1, max=1)
        self.personal_best = np.copy(self.position)
        self.best_fitness = float('inf')

    def update_vel(self, informants_best, global_best):
        """
        updates the velocity based on the formular from https://cs.gmu.edu/~sean/book/metaheuristics/
        :param informants_best:
        :param global_best:
        """
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