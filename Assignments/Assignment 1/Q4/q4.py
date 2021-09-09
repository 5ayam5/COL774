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
    return np.sum(f(Y) * X, axis=0) / np.sum(f(Y))


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
    print(mu(X, Y, lambda y: y), mu(X, Y, lambda y: 1 - y))
