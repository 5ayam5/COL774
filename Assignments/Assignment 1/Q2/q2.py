import numpy as np
import argparse
from os import makedirs
from time import time

parser = argparse.ArgumentParser(description="Program to solve Q1 \
                                  of COL774 Assignment 1")
parser.add_argument('-o', dest='output', type=str, default='./output',
                    help="output directory (if directory does not exist,\
                    it is created) [default './output/']")
parser.add_argument('-s', dest='sample', action='store_true',
                    help="sample data for part a")
parser.add_argument('-l', dest='learn', action='store_true',
                    help="learn using SGD for all batch sizes")
parser.add_argument('-t', dest='test', type=str,
                    help="test set file (to test the trained models)")
args = parser.parse_args()


def sample_data(Theta: np.ndarray, x: np.ndarray, noise_var: float, m: int):
    x = np.insert(x, 0, [1, 0], axis=0)
    X = np.random.normal(x.T[0], np.sqrt(x.T[1]), (m, x.shape[0]))
    Y = np.matmul(X, Theta) + np.random.normal(0, np.sqrt(noise_var), (m, 1))
    return X[:, 1:], Y


def extract_data(folder: str, v: str):
    try:
        # extract data from the csv file
        return np.genfromtxt(folder + '/' + v + '.csv', delimiter=',')
    except IOError:
        return None


# cost function
def cost_function(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray):
    diff = np.matmul(X, Theta) - Y
    return np.matmul(diff.T, diff)[0][0] / (2 * np.shape(X)[0])


# compute the gradient
def gradient(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray):
    return np.matmul(X.T, np.matmul(X, Theta) - Y) / X.shape[0]


# perform stochastic gradient descent
def stochastic_gradient_descent(X: np.ndarray, Y: np.ndarray,
                                eta: float, epsilon: float, r: int,
                                cost=cost_function, grad=gradient):
    eta, epsilon, m = abs(eta), abs(epsilon), X.shape[0]
    X = np.insert(X, 0, np.ones(m), axis=1)
    Theta = np.zeros((X.shape[1], 1))
    j_prev, j = 1 + epsilon, 0
    iters, epochs = 0, 0

    while abs(j - j_prev) > epsilon:
        j_prev, j = j, 0
        for b in range(m // r):
            Xb, Yb = X[b * r:(b + 1) * r], Y[b * r:(b + 1) * r]
            Theta -= eta * grad(Xb, Yb, Theta)
            j += cost(Xb, Yb, Theta)
            iters += 1
        j /= m
        epochs += 1

        if abs(j - j_prev) > 1e10:
            print("Warning: Learning rate too large")
            return None, iters, epochs

    return Theta, iters, epochs


if __name__ == '__main__':
    try:
        makedirs(args.output, exist_ok=True)
    except Exception as e:
        print("error: " + str(e))
        exit(2)

    X, Y = None, None
    # sampling for part a
    if args.sample:
        print("Sampling and saving...")
        X, Y = sample_data(np.array([[3], [1], [2]]),
                           np.array([[3, 4], [-1, 4]]), 2, 1000000)
        np.savetxt(args.output + "/X.csv", X, delimiter=',')
        np.savetxt(args.output + "/Y.csv", Y)
        print("Sample saved!")
    elif args.learn:
        print("Extracting data...")
        X, Y = extract_data(args.output, 'X'), extract_data(args.output, 'Y')
        Y = np.reshape(Y, (X.shape[0], 1))
        perm = np.random.permutation(X.shape[0])
        X, Y = X[perm], Y[perm]
        if X is None or Y is None:
            print("error: could not load data")
            exit(1)
        print("Data extracted!")

    Thetas = np.empty((3, 4))
    times = ["batch size,time taken,iterations,epochs"]
    if args.learn:
        print("Starting learning...")
        for i, r in enumerate([1, 100, 10000, 1000000]):
            t = time()
            Theta, iters, epochs = stochastic_gradient_descent(X, Y, 0.001,
                                                               1e-10, r)
            times.append("{},{},{},{}".format(r, time() - t, iters, epochs))
            Thetas[:, i] = Theta[:, 0]
        np.savetxt(args.output + "/b_thetas.csv", Thetas, delimiter=',')
        with open(args.output + "/b_time.csv", 'w+') as f:
            f.write("\n".join(times))
        print("Learning done!")
    elif args.test is not None:
        print("Extracting learning parameters")
        Thetas = extract_data(args.output, 'b_thetas')
        if (Thetas is None):
            print("error: could not extract learning parameters")
            exit(1)
        print("Learning parameters extracted!")

    if args.test is not None:
        print("Extracting test data...")
        Y = extract_data('.', args.test)
        if Y is None:
            print("error: could not extract test data")
            exit(1)
        print("Test data extracted!\nTesting...")
        Y = Y[1:]
        X = Y[:, :2]
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        Y = Y[:, 2:]
        with open(args.output + "/c.csv", "w+") as f:
            f.write("batch size,error\n")
            for Theta, r in zip(Thetas.T, [1, 100, 10000, 1000000]):
                Theta = np.reshape(Theta, (Theta.shape[0], 1))
                f.write("{},{}\n".format(r, cost_function(X, Y, Theta)))
            f.write("{},{}".format("inf",
                                   cost_function(X, Y,
                                                 np.array([[3], [1], [2]]))))
        print("Testing done!")
