import numpy as np

class Layer:
    def fit(self, X, y, optimizer, loss_function):
        pass
        predicted = self.predict(X)
        loss = loss_function(predicted, y)
        print("loss=", loss)
        loss=predicted-y

        self.backprop(loss, optimizer)

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
        self.input_activation = self.input.predict(data)
        return np.dot(self.W, self.input_activation) + self.b

    def backprop(self, loss, optimizer):
        m = self.input_activation.shape[1]

        dW = np.dot(loss, self.input_activation.T) / m
        db = np.mean(np.sum(loss, axis=1, keepdims=True) / m)

        next_loss = np.dot(self.W.T, loss)
        
        assert (next_loss.shape == self.input_activation.shape)
        assert (dW.shape == self.W.shape)
        assert (isinstance(db, float))

        self.W = self.W - optimizer(dW)
        self.b = self.b - optimizer(db)

        self.input.backprop(next_loss, optimizer)


class InputLayer(Layer):

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def set_data(self, data):
        self.data = data

    def predict(self, data):
        return data

    def backprop(self, loss, optimizer):
        pass
         
class ReluLayer(Layer):

    def __init__(self):
        pass

    def __call__(self, input_layer):
        self.input_dim = input_layer.output_dim
        self.output_dim = self.input_dim
        self.input = input_layer
        return self

    def predict(self, data):
        self.input_activation = self.input.predict(data)
        relu_activation = self.input_activation[:]
        relu_activation[relu_activation<0] = 0
        return relu_activation

    def backprop(self, loss, optimizer):
        next_loss = np.zeros(loss.shape)
        next_loss[self.input_activation > 0] = 1
        self.input.backprop(next_loss * loss, optimizer)

class TanhLayer(Layer):

    def __init__(self):
        pass

    def __call__(self, input_layer):
        self.input_dim = input_layer.output_dim
        self.output_dim = self.input_dim
        self.input = input_layer
        return self

    def predict(self, data):
        self.input_activation = self.input.predict(data)
        return np.tanh(self.input_activation)

    def backprop(self, loss, optimizer):
        next_loss = np.sin(self.input_activation)/np.cos(self.input_activation) 
        self.input.backprop(next_loss * loss, optimizer)

class SoftMaxLayer(Layer):

    def __init__(self):
        pass

    def __call__(self, input_layer):
        self.input_dim = input_layer.output_dim
        self.output_dim = self.input_dim
        self.input = input_layer
        return self

    def predict(self, data):
        self.input_activation = self.input.predict(data)
        e_x = np.exp(self.input_activation - np.max(self.input_activation))
        return e_x / e_x.sum(axis=0)

    def backprop(self, loss, optimizer):
        self.input.backprop(loss, optimizer)



class SumLayer:

    def __init__(self):
        pass

    def __call__(self, input_layer1, input_layer2):
        self.output_dim = min(input_layer1.output_dim, input_layer2.output_dim)
        self.input1 = input_layer1
        self.input2 = input_layer2
        return self

    def predict(self, data):
        self.input_activation1 = self.input1.predict(data)
        self.input_activation2 = self.input2.predict(data)
        return self.input_activation1[:self.output_dim, :] + self.input_activation2[:self.output_dim, :]