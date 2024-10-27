import csv
import numpy as np

class TrainingDataHandler:
    def __init__(self, filename):
        rows = []
        input_values = []
        output_values = []
        with open(filename, 'r') as file:
            csvreader = csv.reader(file)
            fields = next(csvreader)
            for row in csvreader:
                input_values.append(row[:-1])
                output_values.append([row[-1]])

        self.test_data = (input_values[:int(len(input_values) * 0.3)], output_values[:int(len(output_values) * 0.3)])
        self.training_data = (input_values[int(len(input_values) * 0.3):], output_values[int(len(output_values) * 0.3):])

    def get(self):
        return self.training_data, self.test_data


TrainingDataHandler('concrete_data.csv')