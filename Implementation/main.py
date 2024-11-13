from Implementation.Experiment import Experiment

if __name__ == '__main__':
    Experiment.run_experiment_avg('../Data/hyperparameters.json', '../Data/concrete_data.csv', save=True)
