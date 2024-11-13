import json
from matplotlib import pyplot as plt
from Implementation.DataHandler import DataHandler
from Implementation.PSO import PSO
import numpy as np


class Experiment:
    @staticmethod
    def run_experiment_avg(hyper_path=None, train_path=None, save=False, overide_hp = None):
        dh = DataHandler(hyper_path, train_path)
        if overide_hp is not None:
            dh.hyperparameters = overide_hp
        figure = plt.figure(figsize = (25, 12))
        figure.suptitle(f"Total Losses over Time     \n-> {dh.get_title()}")
        i = 0
        all_losses = []
        for row in range(0, 2):
            for col in range(0, 5):
                pso = PSO(dh.hyperparameters, dh.get_training(), dh.get_test())
                pso.debug = True
                best_position, l = pso.optimise()
                pso.test(False)
                ax = plt.subplot2grid((2, 10), (row, col))
                ax.plot(l)
                ax.set_title(f"Run {i} loss : {l[-1]:.3}")
                i+=1
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
        else:
            figure.show()

    @staticmethod
    def run_gridSerach(step):
        i = 0
        total = 3*6*4*6
        for a in np.arange(0.8, 0.95+step, step):
            for b in np.arange(0.3, 0.9+step*2, step*2):
                for g in np.arange(0.3, 0.7+step*2, step*2):
                    for d in np.arange(0, 0.3+step, step):
                        js = json.load(open('../Data/hyperparameters.json', 'r'))
                        js['alpha'] = a
                        js['beta'] = b
                        js['gamma'] = g
                        js['delta'] = d
                        Experiment.run_experiment_avg(train_path = '../Data/concrete_data.csv', overide_hp = js, save = True)
                        i+=1
                        print(f"{i}/{total}")



if __name__ == '__main__':
    Experiment.run_experiment_avg('../Data/hyperparameters.json', '../Data/concrete_data.csv', save=True)
