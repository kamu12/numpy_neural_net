import numpy as np

class Optimizer:

    def __init__(self, lerning_rate):
        self.lerning_rate = lerning_rate

    def __call__(self, gradient):
    	return self.lerning_rate*gradient

class CategoricalCrossentropy:
    def __call__(self, prediction, y):
        m = len(y)

        #print(prediction.shape)
        #print(y.shape)

        #print("prediction=", prediction)
        #print("y=", y)
        #print("np.log(prediction)=", np.log(prediction))
        #print("np.multiply(np.log(prediction), y)=", np.multiply(np.log(prediction), y))
        #print("1 - prediction=", 1 - prediction)
        #print("np.multiply((1 - y), np.log(1 - prediction))=", np.multiply(    (1 - y), np.log(1 - prediction))     )

 
        logprobs = np.multiply(np.log(prediction), y) + np.multiply((1 - y), np.log(1 - prediction))
        cost = - np.sum(logprobs) / m
        return cost
