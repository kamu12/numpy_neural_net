import numpy as np

class Layer:
    def fit(self, X, y, optimizer, loss_function):
        pass
        #predicted = self.__fit_predict(X)
        #loss = loss_function(y, predicted)
        #self.__fit_update(loss, optimizer)

class LiniarLayer(Layer):

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def __call__(self, input_layer):
        self.input_dim = input_layer.output_dim
        self.input = input_layer
        self.W = np.random.randn(self.output_dim, self.input_dim) * np.sqrt(2 / self.input_dim)
        self.b = np.ones((self.output_dim, 1))
        return self

    def predict(self, data):
        input_activation = self.input.predict(data)
        return np.dot(self.W, input_activation) + self.b

class InputLayer(Layer):

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def set_data(self, data):
        self.data = data

    def predict(self, data):
        return data


class ReluLayer(Layer):

    def __init__(self):
        pass

    def __call__(self, input_layer):
        self.input_dim = input_layer.output_dim
        self.output_dim = self.input_dim
        self.input = input_layer
        return self

    def predict(self, data):
        input_activation = self.input.predict(data)
        relu_activation = input_activation[:]# copy
        relu_activation[relu_activation<0] = 0
        return relu_activation

class TanhLayer(Layer):

    def __init__(self):
        pass

    def __call__(self, input_layer):
        self.input_dim = input_layer.output_dim
        self.output_dim = self.input_dim
        self.input = input_layer
        return self

    def predict(self, data):
        input_activation = self.input.predict(data)
        return np.tanh(input_activation)

class SoftMaxLayer(Layer):

    def __init__(self):
        pass

    def __call__(self, input_layer):
        self.input_dim = input_layer.output_dim
        self.output_dim = self.input_dim
        self.input = input_layer
        return self

    def predict(self, data):
        input_activation = self.input.predict(data)
        e_x = np.exp(input_activation - np.max(input_activation))
        return e_x / e_x.sum(axis=0)


class SumLayer:

    def __init__(self):
        pass

    def __call__(self, input_layer1, input_layer2):
        self.output_dim = min(input_layer1.output_dim, input_layer2.output_dim)
        self.input1 = input_layer1
        self.input2 = input_layer2
        return self

    def predict(self, data):
        input_activation1 = self.input1.predict(data)
        input_activation2 = self.input2.predict(data)
        return input_activation1[:self.output_dim, :] + input_activation2[:self.output_dim, :]