import numpy as np
import argparse
from os import makedirs

parser = argparse.ArgumentParser(description="Program to solve Q1 \
                                  of COL774 Assignment 1")
parser.add_argument('input', help="folder containing q4x.dat and q4y.dat")
parser.add_argument('-o', dest='output', type=str, default='./output',
                    help="output directory (if directory does not exist,\
                    it is created) [default './output/']")
args = parser.parse_args()


# normalise matrix
def normalise(X: np.ndarray):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def mu(X: np.ndarray, Y: np.ndarray, f):
    return np.reshape(np.mean(X[np.where(f(Y)), :][0], axis=0), (-1, 1))


def sigma(X: np.ndarray, Y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, f):
    temp = (X - np.where(Y == 0, mu0.T, mu1.T))[np.where(f(Y)), :][0]
    return np.matmul(temp.T, temp) / np.sum(f(Y))


if __name__ == '__main__':
    try:
        makedirs(args.output, exist_ok=True)
    except Exception as e:
        print("error: " + str(e))
        exit(2)

    # extracting data
    try:
        X = np.genfromtxt(args.input + '/q4x.dat', delimiter='  ')
        X = normalise(X)
        Y = np.genfromtxt(args.input + '/q4y.dat',
                          converters={0: lambda s: int(s == b'Alaska')})
        Y = np.reshape(Y, (X.shape[0], 1))
    except IOError:
        print("error: could not load data")
        exit(1)

    phi = sum(Y)[0] / Y.shape[0]
    mu0 = mu(X, Y, lambda y: 1 - y)
    mu1 = mu(X, Y, lambda y: y)
    Sigma = sigma(X, Y, mu0, mu1, lambda y: np.ones(Y.shape))
    Sigma0 = sigma(X, Y, mu0, mu1, lambda y: y == 0)
    Sigma1 = sigma(X, Y, mu0, mu1, lambda y: y == 1)
    with open(args.output + "/a", 'w+') as out:
        out.write("\n".join(["phi    = " + str(phi),
                             "mu_0   = " + str(mu0.T),
                             "mu_1   = " + str(mu1.T),
                             "sigma  = ", str(Sigma),
                             "sigma0 = ", str(Sigma0),
                             "sigma1 = ", str(Sigma1)]))
