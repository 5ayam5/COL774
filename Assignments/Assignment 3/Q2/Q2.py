from os import makedirs
from time import time
import argparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt


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
                 learning_rate=0.1, epsilon=1e-8, batch_size=100, max_epochs=1000, adaptive=False, verbose=False):
        permutation = np.random.permutation(X.shape[0])
        self.X = X[permutation]
        self.Y = Y[permutation]
        self.layers = [X.shape[1]] + layers + [Y.shape[1]]
        self.weights = []
        self.biases = []
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.learning_rate = learning_rate
        if adaptive:
            self.learning_rate *= 10
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.adaptive = adaptive
        self.verbose = verbose
        self.init_weights_biases()

    def init_weights_biases(self):
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(
                self.layers[i], self.layers[i - 1]) * (2 / self.layers[i - 1]) ** 0.5)
            self.biases.append(np.random.randn(
                self.layers[i], 1) * (2 / self.layers[i - 1]) ** 0.5)

    def forward_propagation(self, X):
        self.a = [X]
        for i in range(len(self.layers) - 1):
            z = np.matmul(self.a[i], self.weights[i].T) + self.biases[i].T
            if i == len(self.layers) - 2:
                self.a.append(sigmoid(z))
            else:
                self.a.append(self.activation(z))
        return self.a[-1]

    def back_propagation(self, Y):
        self.delta = [(Y - self.a[-1]) *
                      sigmoid_derivative(self.a[-1])]
        for i in range(len(self.layers) - 2, 0, -1):
            self.delta.append(
                np.matmul(self.delta[-1], self.weights[i]) * self.activation_derivative(self.a[i]))
        self.delta.reverse()

    def update_weights_biases(self, i):
        rate = self.learning_rate
        if self.adaptive:
            rate = self.learning_rate / i ** 0.5
        for i in range(len(self.layers) - 1):
            self.weights[i] += rate * \
                np.matmul(self.delta[i].T, self.a[i]) / self.batch_size
            self.biases[i] += rate * \
                np.sum(self.delta[i], axis=0,
                       keepdims=True).T / self.batch_size

    def train(self):
        prev_cost, curr_cost, i = 1e9, 0, 0
        while abs(curr_cost - prev_cost) > self.epsilon and i < self.max_epochs:
            prev_cost = curr_cost
            for j in range(0, len(self.X), self.batch_size):
                X = self.X[j:j + self.batch_size]
                Y = self.Y[j:j + self.batch_size]
                self.forward_propagation(X)
                self.back_propagation(Y)
                self.update_weights_biases(i + 1)
                curr_cost += self.cost(X, Y)
            curr_cost /= len(self.X) / self.batch_size
            i += 1
            if self.verbose:
                print(f'Epoch: {i}, Cost: {curr_cost}')
                if i % 100 == 0:
                    print(f'Accuracy: {self.accuracy(self.X, self.Y)}')

    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)

    def cost(self, X, Y):
        return np.sum(np.square(Y - self.forward_propagation(X))) / (2 * self.batch_size)

    def accuracy(self, X, Y):
        return np.sum(self.predict(X) == np.argmax(Y, axis=1)) / len(Y)

    def confusion_matrix(self, X, Y):
        Y_pred = self.predict(X)
        Y = np.argmax(Y, axis=1)
        k = self.Y.shape[1]
        confusion = np.zeros((k, k), dtype=int)
        for y, y_pred in zip(Y, Y_pred):
            confusion[y, y_pred] += 1
        return confusion


def util_nn(part: str, params, activation, activation_derivative, adaptive: bool = False):
    with open(f'{args.output}/{part}', 'w+') as f:
        f.write('units,train_accuracy,test_accuracy,time\n')
        for num_units in params:
            nn = NeuralNetwork(X_train, Y_train, num_units, activation,
                               activation_derivative, adaptive=adaptive, verbose=True)
            t = time()
            nn.train()
            t = time() - t
            f.write(
                f'{num_units[0]},{nn.accuracy(X_train, Y_train)},{nn.accuracy(X_test, Y_test)},{t}\n')
            with open(f'{args.output}/{part}_{num_units[0]}_confusion', 'w+') as g:
                g.write(str(nn.confusion_matrix(X_test, Y_test)))

    if part[0] != 'e':
        df = pd.read_csv(f'{args.output}/{part}')
        df.plot(x='units', y=['train_accuracy',
                'test_accuracy'], title='Accuracy')
        plt.savefig(f'{args.output}/{part}_accuracy.png')
        plt.clf()
        df.plot(x='units', y='time', title='Time')
        plt.savefig(f'{args.output}/{part}_time.png')


def train_MLP(layers, architecture):
    nn = MLPClassifier(layers, architecture, solver='sgd',
                       learning_rate_init=0.1, max_iter=1000)
    nn.fit(X_train, Y_train)
    return nn


def util_MLP():
    with open(f'{args.output}/MLP', 'w+') as f:
        f.write('architecture,train_accuracy,test_accuracy,time\n')
        t = time()
        nn = train_MLP([100, 100], 'relu')
        t = time() - t
        f.write(
            f'relu,{nn.score(X_train, Y_train)},{nn.score(X_test, Y_test)},{t}\n')
        f.write(
            f'{confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(nn.predict(X_test), axis=1))}\n')
        t = time()
        nn = train_MLP([100, 100], 'logistic')
        t = time() - t
        f.write(
            f'sigmoid,{nn.score(X_train, Y_train)},{nn.score(X_test, Y_test)},{t}\n')
        f.write(
            f'{confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(nn.predict(X_test), axis=1))}\n')

    return None


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
        util_nn('c', [[5], [10], [15], [20], [25]],
                sigmoid, sigmoid_derivative)

    if args.question.find('d') != -1:
        util_nn('d', [[5], [10], [15], [20], [25]],
                sigmoid, sigmoid_derivative, adaptive=True)

    if args.question.find('e') != -1:
        util_nn('e_relu', [[100, 100]], relu, relu_derivative, adaptive=True)
        util_nn('e_sigmoid', [[100, 100]], sigmoid,
                sigmoid_derivative, adaptive=True)

    if args.question.find('f') != -1:
        util_MLP()
