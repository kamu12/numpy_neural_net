from nn_classes import *
from utils import *
from data_load import load_data

if __name__ == "__main__":
    np.random.seed(42)
    X,true_val = load_data("../data/spambase/spambase.data")

    y = np.zeros((len(true_val), 2))
    y[:, 0]=true_val
    y[:, 1]=np.abs(true_val-1)

    X = np.array([[-1, -1, 3], [1, 2, 3]]).T
    y = np.array([[0, 1], [1, 0]])#.reshape((1, 2))
    """
    input_data = InputLayer(3)

    model = LiniarLayer(2)(input_data)
    model = SoftMaxLayer()(model)
    sg = Optimizer(0.01)
    categorical_crossentropy = CategoricalCrossentropy()
    
    for i in range(10):
        model.fit(X, y, sg, categorical_crossentropy)
    
    """

    input_data = InputLayer(3)

    model = LiniarLayer(10)(input_data)
    model = TanhLayer()(model)
    
    model = LiniarLayer(5)(model)
    #model = SumLayer()(input_data, model)
    model = ReluLayer()(model)
    
    model = LiniarLayer(2)(model)
    model = SoftMaxLayer()(model)

    X = np.array([[-1, -1, 3], [1, 2, 3]]).T
    y = np.array([0, 1]).T
    res = model.predict(X)
    #print(res.shape)
    #print(res)

    sg = Optimizer(0.001)
    categorical_crossentropy = CategoricalCrossentropy()

    for i in range(1):
        model.fit(X, y, sg, categorical_crossentropy)
        predicted = model.predict(X)
        print(predicted[0])
        predicted_classes = np.round(predicted)
        print(predicted_classes[0])
        print("accuracy ", np.mean(true_val == predicted[:, 0]))
    

    
