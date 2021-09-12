import numpy as np
import matplotlib.pyplot as plt
import argparse
from os import makedirs

parser = argparse.ArgumentParser(description="Program to solve Q1 \
                                  of COL774 Assignment 1")
parser.add_argument('input', help="folder containing q4x.dat and q4y.dat")
parser.add_argument('-o', dest='output', type=str, default='./output',
                    help="output directory (if directory does not exist,\
                    it is created) [default './output/']")
parser.add_argument('-d', dest='display', action='store_true',
                    help="display and animate plot for different subparts")
args = parser.parse_args()


# normalise matrix
def normalise(X: np.ndarray):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def mu(X: np.ndarray, Y: np.ndarray, f):
    return np.reshape(np.mean(X[np.where(f(Y)), :][0], axis=0), (-1, 1))


def sigma(X: np.ndarray, Y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, f):
    temp = (X - np.where(Y == 0, mu0.T, mu1.T))[np.where(f(Y)), :][0]
    return np.matmul(temp.T, temp) / np.sum(f(Y))


def gda_linear(x: np.ndarray, phi: float, mu0: np.ndarray, mu1: np.ndarray,
               Sigma_inv: np.ndarray):
    return (np.log((1 - phi) / phi)
            - np.matmul(np.matmul((mu1 - mu0).T, Sigma_inv), x)
            + ((np.matmul(np.matmul(mu1.T, Sigma_inv), mu1)
                + np.matmul(np.matmul(mu0.T, Sigma_inv), mu1))) / 2)[0][0]


def gda_general(x: np.ndarray, phi: float, mu0: np.ndarray, mu1: np.ndarray,
                Sigma0_inv: np.ndarray, Sigma1_inv: np.ndarray):
    return (np.log(np.sqrt(np.linalg.det(Sigma0_inv)
                           / np.linalg.det(Sigma1_inv))
                   * (1 - phi) / phi)
            - np.matmul(np.matmul(mu1.T, Sigma1_inv)
                        - np.matmul(mu0.T, Sigma0_inv), x)
            + np.matmul(np.matmul(x.T, Sigma1_inv - Sigma0_inv), x) / 2
            + (np.matmul(np.matmul(mu1.T, Sigma1_inv), mu1)
               - np.matmul(np.matmul(mu0.T, Sigma0_inv), mu0)) / 2)[0][0]


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
    with open(args.output + "/ad", 'w+') as out:
        out.write("\n".join(["phi    = " + str(phi),
                             "mu_0   = " + str(mu0.T),
                             "mu_1   = " + str(mu1.T),
                             "sigma  = ", str(Sigma),
                             "sigma0 = ", str(Sigma0),
                             "sigma1 = ", str(Sigma1)]))

    fig, axes = plt.subplots()
    axes.set_title('Classification of Salmons')
    axes.set_xlabel('$x_1$ (diameter in freshwater)')
    axes.set_ylabel('$x_2$ (diameter in marine water)')
    axes.scatter(*X[np.where(Y == 1), :][0].T, c='blue',
                 marker='x', label='Alaska')
    axes.scatter(*X[np.where(Y == 0), :][0].T, c='green',
                 marker='o', label='Canada')

    x1, x2 = np.meshgrid(*np.linspace([-2, -2], [2, 2], 10).T)
    z = np.apply_along_axis(lambda x: gda_linear(x, phi, mu0, mu1,
                            np.linalg.inv(Sigma)), 2,
                            np.stack([x1, x2], axis=-1))
    axes.contour(x1, x2, z, 0, colors='red')
    axes.plot([], [], color='red', label='linear separator')
    z = np.apply_along_axis(lambda x: gda_general(x, phi, mu0, mu1,
                            np.linalg.inv(Sigma0), np.linalg.inv(Sigma1)),
                            2, np.stack([x1, x2], axis=-1))
    axes.contour(x1, x2, z, 0, colors='orange')
    axes.plot([], [], color='orange', label='quadratic separator')
    axes.legend()
    if args.display:
        plt.show()
    fig.savefig(args.output + "/bce.png")
    plt.close()
