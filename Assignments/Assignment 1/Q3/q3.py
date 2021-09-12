import numpy as np
import matplotlib.pyplot as plt
import argparse
from os import makedirs

parser = argparse.ArgumentParser(description="Program to solve Q1 \
                                  of COL774 Assignment 1")
parser.add_argument('input', help="folder containing logisticX.csv\
                    and logisticY.csv")
parser.add_argument('-o', dest='output', type=str, default='./output',
                    help="output directory (if directory does not exist,\
                    it is created) [default './output/']")
parser.add_argument('-d', dest='display', action='store_true',
                    help="display the plot")
args = parser.parse_args()


# read data from csv file
def extract_data(v: str):
    try:
        # return column vector of the csv file
        return np.genfromtxt(args.input + '/logistic' + v + '.csv',
                             delimiter=',')
    except IOError:
        return None


# normalise matrix
def normalise(X: np.ndarray):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# compute the hessian matrix
def hessian(X: np.ndarray, theta: np.ndarray):
    temp = np.exp(-np.matmul(X, theta).T)
    return np.matmul(X.T * temp / np.square(1 + temp), X)


# gradient
def gradient(X: np.ndarray, Y: np.ndarray, theta: np.ndarray):
    diff = Y - 1 / (1 + np.exp(-np.matmul(X, theta)))
    return np.matmul(X.T, diff)


# cost function
def cost_function(X: np.ndarray, Y: np.ndarray, theta: np.ndarray):
    h_theta = 1 / (1 + np.exp(-np.matmul(X, theta)))
    return (np.matmul(Y.T, np.log(h_theta)) +
            np.matmul(1 - Y.T, np.log(1 - h_theta)))[0][0]


def newton_method(X: np.ndarray, Y: np.ndarray, epsilon: float,
                  cost=cost_function, grad=gradient, hessian=hessian):
    m, epsilon = X.shape[0], abs(epsilon)
    X = np.insert(X, 0, np.ones(m), axis=1)
    theta = np.zeros((X.shape[1], 1))
    j, j_prev = 1 + epsilon, 0
    i = 0

    while abs(j - j_prev) > epsilon:
        theta = theta + np.matmul(np.linalg.inv(hessian(X, theta)),
                                  grad(X, Y, theta))
        j_prev, j = j, cost(X, Y, theta)
        i += 1

    return theta, i


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
    Y = np.reshape(Y, (X.shape[0], 1))
    X = normalise(X)

    # learn model
    epsilon = 1e-20
    theta, iterations = newton_method(X, Y, epsilon)

    with open(args.output + "/a", 'w+') as out:
        out.write("\n".join(["              " + str(theta[0][0]),
                             "theta       = " + str(theta[1][0]),
                             "              " + str(theta[2][0]),
                             "#iterations = " + str(iterations),
                             "epsilon     = " + str(epsilon)]))

    fig, axes = plt.subplots()
    axes.set_title('Logistic Regression')
    axes.set_xlabel('x_1')
    axes.set_ylabel('x_2')
    axes.scatter(X[Y.T[0] == 0].T[0], X[Y.T[0] == 0].T[1], c='blue',
                 marker='x', label='class 1')
    axes.scatter(X[Y.T[0] == 1].T[0], X[Y.T[0] == 1].T[1], c='green',
                 marker='o', label='class 2')
    axes.plot(X.T[0], -(theta[0][0] + theta[1][0] * X.T[0]) / theta[2][0],
              c='red', label='separator')
    axes.legend()
    if args.display:
        plt.show()
    fig.savefig(args.output + "/b.png")
    plt.close()
