from nn_classes import *
from utils import *
from data_load import load_data
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def accuracy(pred, y):
    pred_indexes = np.argmax(predicted, axis=0)
    actual_indexes = np.argmax(y, axis=0)
    return np.count_nonzero(pred_indexes == actual_indexes) / y.shape[1]


if __name__ == "__main__":
    np.random.seed(42)
    minibatch_size = 20
    history = []
    X, true_val = load_data("../data/spambase/spambase.data")

    y = np.zeros((len(true_val), 2))
    y[:, 0] = true_val
    y[:, 1] = np.abs(true_val - 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T

    input_data = InputLayer(X.shape[1])

    model = LiniarLayer(100)(input_data)
    model = TanhLayer()(model)

    model = LiniarLayer(30)(model)
    # model = SumLayer()(input_data, model)
    model = ReluLayer()(model)

    model = LiniarLayer(2)(model)
    model = SoftMaxLayer()(model)

    sg = Optimizer(0.001)
    categorical_crossentropy = CategoricalCrossentropy()

    for i in range(15):
        np.random.shuffle(X_train)
        np.random.shuffle(y_train)

        for i in range(0, X_train.shape[1], minibatch_size):
            X_train_mini = X_train[:, i:i + minibatch_size]
            y_train_mini = y_train[:, i:i + minibatch_size]

            model.fit(X_train_mini, y_train_mini, sg, categorical_crossentropy)

        predicted = model.predict(X_train)
        acc = accuracy(predicted, y_train)
        history.append(acc)
        print("train accuracy: ", acc)

    predicted = model.predict(X_test)
    acc = accuracy(predicted, y_test)
    print("test accuracy: ", acc)




