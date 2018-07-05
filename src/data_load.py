import numpy as np


def load_data(spambase_path):
	spambase_dataset = np.loadtxt(open(spambase_path, 'rb'), delimiter=',')

	spambase_data = spambase_dataset[:, list(range(0, spambase_dataset.shape[1] - 1))]
	spambase_labels = spambase_dataset[:, spambase_dataset.shape[1] - 1]
	return spambase_data, spambase_labels

if __name__ == "__main__":
	X, Y = load_data("../data/spambase/spambase.data")
	print(X.shape)
	print("first row ", X[0])
	print(Y.shape)
	print("first row ", Y[0])