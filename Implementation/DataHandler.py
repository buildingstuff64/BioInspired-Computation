import json
import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, hyper_path, train_path):
        if hyper_path != None:
            self.hyperparameters = json.load(open(hyper_path, 'r'))

        if train_path == "":
            return
        file = pd.read_csv(train_path)
        input_data = np.array(file.iloc[:, : 8].to_numpy())
        output_data = np.array(file.iloc[:, 8:].to_numpy())

        self.input_data_train = input_data[int(len(input_data) * 0.7):]
        self.output_data_train = output_data[int(len(output_data) * 0.7):]

        self.input_data_test = input_data[-int(len(input_data) * 0.3):]
        self.output_data_test = output_data[-int(len(output_data) * 0.3):]


    def get_training(self):
        return self.input_data_train, self.output_data_train

    def get_test(self):
        return self.input_data_test, self.output_data_test

    def get_str_name(self, h=None):
        if h is None:
            h = self.hyperparameters
        str = f'{h['swarm_size']}_{h['iterations']:d}_{h['activation']}_{h['informants']}_{h['alpha']}_{h['beta']}_{h['gamma']}_{h['delta']}'
        return str.replace(".", "")

    def get_title(self, h=None):
        if h is None:
            h = self.hyperparameters
        return f'swarm size:{h['swarm_size']}   iterations:{h['iterations']:d}  activation:{h['activation']}    informants:{h['informants']}        a:{h['alpha']}  b:{h['beta']}   g:{h['gamma']}  d:{h['delta']}'
