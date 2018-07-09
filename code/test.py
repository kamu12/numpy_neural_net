from nn_classes import *
from utils import *
from data_load import load_data

def accuracy(pred, y):
    pred_indexes = np.argmax(predicted, axis=0)
    actual_indexes = np.argmax(y, axis=0)
    return np.count_nonzero(pred_indexes == actual_indexes) / y.shape[1]

if __name__ == "__main__":
    np.random.seed(42)
    X,true_val = load_data("../data/spambase/spambase.data")

    y = np.zeros((len(true_val), 2))
    y[:, 0]=true_val
    y[:, 1]=np.abs(true_val-1)


    input_data = InputLayer(X.shape[1])

    model = LiniarLayer(100)(input_data)
    model = TanhLayer()(model)
    
    model = LiniarLayer(30)(model)
    #model = SumLayer()(input_data, model)
    model = ReluLayer()(model)
    
    model = LiniarLayer(2)(model)
    model = SoftMaxLayer()(model)

    sg = Optimizer(0.001)
    categorical_crossentropy = CategoricalCrossentropy()

    X = X.T
    y = y.T
    for i in range(100):
        model.fit(X, y, sg, categorical_crossentropy)
        predicted = model.predict(X)
        print("accuracy ", accuracy(predicted, y))
    

    
