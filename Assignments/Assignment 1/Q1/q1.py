import numpy as np
import argparse
from os import makedirs

parser = argparse.ArgumentParser(description="Program to solve Q1 \
                                  of COL774 Assignment 1")
parser.add_argument('input', help="folder containing linearX.csv\
                    and linearY.csv")
parser.add_argument('-o', dest='output', type=str, default='./output',
                    help="output directory (if directory does not exist,\
                    it is created) [default './output/']")
args = parser.parse_args()


# read data from csv file
def extract_data(v: str):
    try:
        # return column vector of the csv file
        return np.loadtxt(open(args.input + 'linear' + v + '.csv'), ndmin=2)
    except IOError:
        return None


# normalise matrix
def normalise(X: np.ndarray):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# cost function
def cost_function(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray):
    diff = np.matmul(X, Theta) - Y
    return np.matmul(diff.T, diff)[0][0] / (2 * np.shape(X)[0])


# compute the gradient
def gradient(X: np.ndarray, Y: np.ndarray, Theta: np.ndarray):
    return np.matmul(X.T, np.matmul(X, Theta) - Y) / X.shape[0]


# perform gradient descent
def gradient_descent(X: np.ndarray, Y: np.ndarray, eta: float, epsilon: float,
                     cost=cost_function, grad=gradient):
    eta, epsilon = abs(eta), abs(epsilon)
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    Theta = np.array([[0.], [0.]])
    j_prev, j = 1 + epsilon, 0
    i = 0
    while abs(j - j_prev) > epsilon:
        j_prev = j
        Theta -= eta * grad(X, Y, Theta)
        j = cost(X, Y, Theta)
        i += 1
    return Theta, i


if __name__ == '__main__':
    try:
        makedirs(args.output, exist_ok=True)
    except Exception as e:
        print("error: " + str(e))
        exit(2)

    # extracting data
    X, Y = extract_data('X'), extract_data('Y')
    if X is None or Y is None:
        print("error: could not load data")
        exit(1)
    X = normalise(X)

    # code for part a
    eta, epsilon = 1e-2, 1e-15
    Theta, iterations = gradient_descent(X, Y, eta, epsilon)
    with open(args.output + "/a", "w+") as out:
        out.writelines("\n".join(["learning rate\t\t= " + str(eta),
                                  "epsilon\t\t\t= " + str(epsilon),
                                  "[theta0, theta1]\t= [" + str(Theta[0][0]) +
                                  ", " + str(Theta[1][0]) + "]",
                                  "#iterations\t\t= " + str(iterations)]))
