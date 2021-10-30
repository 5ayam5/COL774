import numpy as np
from sklearn.preprocessing import OneHotEncoder
import argparse
from os import makedirs
from time import time


# function to parse question part, data location
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-question', type=str,
                        default='acdef', help='Part number')
    parser.add_argument('-train_data', type=str,
                        default='data/poker-hand-training.data', help='Training data')
    parser.add_argument('-test_data', type=str,
                        default='data/poker-hand-testing.data', help='Test data')
    parser.add_argument('-one_hot_encoded', action='store_true')
    parser.add_argument('-output', type=str,
                        default='./output', help='Output directory')
    args = parser.parse_args()
    makedirs(args.output, exist_ok=True)
    return args


def one_hot_encoder(arr: np.ndarray):
    enc = OneHotEncoder()
    return enc.fit_transform(arr).toarray()


def extract_data(filename: str):
    data = np.loadtxt(filename, delimiter=',')
    X, Y = data[:, :-1], data[:, -1]
    Y = Y.reshape((-1, 1))
    return X, Y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return 1. * (z > 0)


class NeuralNetwork:
    def __init__(self, X, Y, layers, activation, activation_derivative,
                 learning_rate=1, epochs=1000, batch_size=100, verbose=False):
        self.X = X
        self.Y = Y
        self.layers = [X.shape[1]] + layers + [Y.shape[1]]
        self.weights = []
        self.biases = []
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.init_weights_biases()

    def init_weights_biases(self):
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(
                self.layers[i], self.layers[i - 1]))
            self.biases.append(np.random.randn(self.layers[i], 1))

    def forward_propagation(self, X):
        self.a = [X]
        for i in range(len(self.layers) - 1):
            z = np.matmul(self.a[i], self.weights[i].T) + self.biases[i].T
            self.a.append(self.activation(z))
        return self.a[-1]

    def back_propagation(self, Y):
        self.delta = [(Y - self.a[-1]) *
                      self.activation_derivative(self.a[-1])]
        for i in range(len(self.layers) - 2, 0, -1):
            self.delta.append(
                np.dot(self.delta[-1], self.weights[i]) * self.activation_derivative(self.a[i]))
        self.delta.reverse()

    def update_weights_biases(self):
        for i in range(len(self.layers) - 1):
            self.weights[i] += self.learning_rate * \
                np.dot(self.delta[i].T, self.a[i]) / self.batch_size
            self.biases[i] += self.learning_rate * \
                np.sum(self.delta[i], axis=0,
                       keepdims=True).T / self.batch_size

    def train(self):
        for i in range(self.epochs):
            for j in range(0, len(self.X), self.batch_size):
                self.forward_propagation(self.X[j:j + self.batch_size])
                self.back_propagation(self.Y[j:j + self.batch_size])
                self.update_weights_biases()
            if self.verbose:
                print(f'Epoch: {i}, Cost: {self.cost()}')
                if i % 100 == 0:
                    print(f'Accuracy: {self.accuracy(self.X, self.Y)}')

    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)

    def cost(self):
        return np.sum(np.square(self.Y - self.forward_propagation(self.X))) / (2 * self.batch_size)

    def accuracy(self, X, Y):
        return np.sum(self.predict(X) == np.argmax(Y, axis=1)) / len(Y)


if __name__ == "__main__":
    args = parse_args()
    if args.question.find('a') != -1:
        X_train, Y_train = extract_data(args.train_data)
        X_train, Y_train = one_hot_encoder(
            X_train), one_hot_encoder(Y_train)
        X_test, Y_test = extract_data(args.test_data)
        X_test, Y_test = one_hot_encoder(X_test), one_hot_encoder(Y_test)
        np.save(f'{args.output}/X_test.npy', X_test)
        np.save(f'{args.output}/Y_test.npy', Y_test)
        np.save(f'{args.output}/X_train.npy', X_train)
        np.save(f'{args.output}/Y_train.npy', Y_train)
    else:
        X_test = np.load(f'{args.output}/X_test.npy')
        Y_test = np.load(f'{args.output}/Y_test.npy')
        X_train = np.load(f'{args.output}/X_train.npy')
        Y_train = np.load(f'{args.output}/Y_train.npy')

    if args.question.find('c') != -1:
        params = [5, 10, 15, 20, 25]
        with open(f'{args.output}/c', 'w+') as f:
            f.write('units,train_accuracy,test_accuracy,time\n')
            for num_units in params:
                nn = NeuralNetwork(X_train, Y_train, [num_units], sigmoid, sigmoid_derivative)
                t = time()
                nn.train()
                t = time() - t
                f.write(f'{num_units},{nn.accuracy(X_train, Y_train)},{nn.accuracy(X_test, Y_test)},{t}\n')
