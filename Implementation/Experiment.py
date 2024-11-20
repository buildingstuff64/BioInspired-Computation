import json
import random

from matplotlib import pyplot as plt
from pandas.compat.numpy import np_long

from Implementation.DataHandler import DataHandler
from Implementation.PSO import PSO
import numpy as np

class best:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.g = 0
        self.d = 0
        self.best = 1000

    def set(self, ss, it, a, b, c, d,inf, best):
        self.a = a
        self.b = b
        self.g = c
        self.d = d
        self.best = best
        self.swarm_size = ss
        self.iter = it
        self.info = inf
        print(self)
        f = open('best_loss.txt', "a")
        f.write(self.__str__())
        f.close()

    def __str__(self):
        return f"\n swarm_size = {self.swarm_size}, iterations = {self.iter}, informants = {self.info}, a = {self.a}, b = {self.b}, g = {self.g}, d = {self.d} \n best = {self.best}"


class Experiment:
    @staticmethod
    def run_experiment_avg(hyper_path=None, train_path=None, save=False, overide_hp = None, show=True):
        dh = DataHandler(hyper_path, train_path)
        if overide_hp is not None:
            dh.hyperparameters = overide_hp
        figure = plt.figure(figsize = (25, 12))
        figure.suptitle(f"Total Losses over Time     \n-> {dh.get_title()}")
        i = 0
        all_losses = []
        for row in range(0, 2):
            for col in range(0, 5):
                l = Experiment.run_single_experiment(dh)
                ax = plt.subplot2grid((2, 10), (row, col))
                ax.plot(l)
                ax.set_title(f"Run {i} loss : {l[-1]:.3}")
                i += 1
                all_losses.append(l)

        average_over_all = []
        length = len(all_losses)
        for i in range(0, len(all_losses[0])):
            avg = []
            for x in all_losses:
                avg.append(x[i])
            average_over_all.append(sum(avg)/length)


        final = plt.subplot2grid((2, 10), (0, 5), rowspan = 2, colspan = 5)
        final.plot(average_over_all)
        final.set_title(f"Average Loss of all Runs   loss : {average_over_all[-1]:.3}")
        figure.tight_layout()
        if save:
            figure.savefig(f'../Experiments/{dh.get_str_name()}.png')
        if show:
            plt.show()
        return average_over_all[-1]

    @staticmethod
    def run_single_experiment(dh):
        pso = PSO(dh.hyperparameters, dh.get_training(), dh.get_test())
        pso.debug = True
        best_position, l = pso.optimise()
        pso.test(False)
        return l

    @staticmethod
    def run_gridSerach(step):
        i = 0
        total = len(np.arange(0.8, 0.95+step, step))*len(np.arange(0.3, 0.9+step*2, step*2))*len(np.arange(0.3, 0.7+step*2, step*2))*len(np.arange(0, 0.3+step, step))

        BEST = best()
        for a in np.arange(0.8, 0.95+step, step):
            for b in np.arange(0.3, 0.9+step*2, step*2):
                for g in np.arange(0.3, 0.7+step*2, step*2):
                    for d in np.arange(0, 0.3+step, step):
                        js = json.load(open('../Data/hyperparameters.json', 'r'))

                        js['alpha'] = a
                        js['beta'] = b
                        js['gamma'] = g
                        js['delta'] = d
                        dh = DataHandler(None, '../Data/concrete_data.csv')
                        dh.hyperparameters = js
                        loss = Experiment.run_single_experiment(dh)
                        #avg = Experiment.run_experiment_avg(train_path = '../Data/concrete_data.csv', overide_hp = js, save = False, show=False)
                        if loss[-1] < BEST.best:
                            BEST.set(a, b, g, d, loss[-1])
                        i+=1
                        print(f"{i}/{total}")
        print(BEST)

    @staticmethod
    def run_random_search(count, budget):
        BEST = best()
        js = json.load(open('../Data/hyperparameters.json', 'r'))
        for i in range(0, count):
            div = random.uniform(0.01, 1)
            js['swarm_size'] = int(budget * div)
            js['iterations'] = int(budget * (1-div))
            js['informants'] = int(js['swarm_size'] * random.uniform(0, 0.25))+1

            js['alpha'] = random.uniform(0.7, 0.95)
            js['beta'] = random.uniform(0.5, 0.9)
            js['gamma'] = random.uniform(0.1, 0.5)
            js['delta'] = random.uniform(0.01, 0.2)

            dh = DataHandler(None, '../Data/concrete_data.csv')
            dh.hyperparameters = js
            loss = Experiment.run_single_experiment(dh)

            if loss[-1] < BEST.best:
                BEST.set(js['swarm_size'], js['iterations'], js['alpha'], js['beta'], js['gamma'], js['delta'], js['informants'] ,loss[-1])

            if loss[-1] < BEST.best + 0.1:
                print("close so runnning moreeee")
                for i in range(0, 10):
                    loss = Experiment.run_single_experiment(dh)
                    if loss[-1] < BEST.best:
                        BEST.set(js['swarm_size'], js['iterations'], js['alpha'], js['beta'], js['gamma'], js['delta'], js['informants'], loss[-1])





if __name__ == '__main__':
    #Experiment.run_experiment_avg('../Data/hyperparameters.json', '../Data/concrete_data.csv', save=True)
    Experiment.run_random_search(1000, 150)
