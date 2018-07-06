import numpy as np

class Optimizer:

    def __init__(self, lerning_rate):
        self.lerning_rate = lerning_rate

class CategoricalCrossentropy:
    def __call__(self, prediction, y):
        m = y.shape[1]
        logprobs = np.multiply(np.log(prediction), y) +\
                   np.multiply((1 - y), np.log(1 - prediction))
        cost = - np.sum(logprobs) / m
        return cost
