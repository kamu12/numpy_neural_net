from nn_classes import *
from utils import *
from data_load import load_data

if __name__ == "__main__":
    np.random.seed(42)
    X,y = load_data("../data/spambase/spambase.data")

    input_data = InputLayer(3)

    model = LiniarLayer(10)(input_data)
    model = TanhLayer()(model)
    
    model = LiniarLayer(5)(model)
    model = SumLayer()(input_data, model)
    model = ReluLayer()(model)
    
    model = LiniarLayer(3)(model)
    model = SoftMaxLayer()(model)

    X = np.array([[-1, -1, 3], [1, 2, 3]]).T
    y = np.array([0, 1]).T
    res = model.predict(X)
    print(res.shape)
    print(res)

    sg = Optimizer(0.01)
    categorical_crossentropy = CategoricalCrossentropy()
    model.fit(X, y, sg, categorical_crossentropy)

    
