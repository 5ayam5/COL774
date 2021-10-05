from q2_util import *
import argparse
from os import makedirs
from time import time


def get_args():
    parser = argparse.ArgumentParser(description="Q2 of COL774 Assignment")
    parser.add_argument('train', type=str, help="path of the training data")
    parser.add_argument('test', type=str, help="path of test data")
    parser.add_argument(
        'type', type=int, help="type of classifier: binary (0) or multi (1)")
    parser.add_argument('part', type=str, help="which part number")
    parser.add_argument('-d', dest='digit', type=int, default=9,
                        help="first digit for binary classification")
    parser.add_argument('-o', dest='output', type=str,
                        default='./output', help="output directory")
    parser.add_argument('-c', dest='c', type=float,
                        default=1.0, help="parameter for soft SVMs")
    parser.add_argument('-g', dest='gamma', type=float,
                        default=0.05, help="gamma for Gaussian kernel")
    parser.add_argument('-e', dest='tol', type=float,
                        default=1e-4, help="tolerance for alpha")
    parser.add_argument('-p', dest='pick', action='store_true',
                        help="extract model from a pickle dump")

    args = parser.parse_args()
    makedirs(args.output, exist_ok=True)
    return args


def cvxopt_util(args: argparse.Namespace, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, part: str, prod, pred, gamma: float = None):
    print("Training model...")
    t = time()

    indices, alpha, b = svm_cvxopt(
        X_train, Y_train, args.c, args.tol, prod, gamma)
    support_X = X_train[indices]
    support_Y = Y_train[indices]
    w = None
    if part == 'linear':
        w = np.matmul((support_Y * support_X).T, alpha)
        def prediction(X): return pred(w, b, X)
    else:
        def prediction(X): return pred(alpha, support_Y, support_X, b, X)

    t = time() - t
    print("Model trained!\nMaking predictions and writing to file...")

    with open(args.output + '/bin_cvxopt_' + part, 'w+') as f:
        f.write("training time         = {}\n".format(t))
        f.write("nSV                   = {}\n".format(len(indices)))
        f.write("b                     = {}\n".format(b))
        f.write("accuracy(train, test) = {}".format(accuracy_util_cvxopt(
            X_train, Y_train, X_test, Y_test, prediction)))
        if part == 'linear':
            f.write("\nw =\n{}".format(w))

    print("Written to {}!".format(args.output + '/bin_cvxopt_' + part))


def libsvm_util(args: argparse.Namespace, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, part: str, compute_model):
    print("Training {} model...".format(part))
    t = time()

    indices, alpha, model, b = compute_model()
    w = None
    if part == 'linear':
        support_X = X_train[indices]
        support_Y = Y_train[indices]
        w = np.matmul((support_Y * support_X).T, alpha)

    t = time() - t
    print("Model trained!\nMaking predictions and writing to file...")

    with open(args.output + '/bin_libsvm_' + part, 'w+') as f:
        f.write("training time         = {}\n".format(t))
        f.write("nSV                   = {}\n".format(model.get_nr_sv()))
        f.write("b                     = {}\n".format(b))
        f.write("accuracy(train, test) = {}".format(
            accuracy_util_libsvm(X_train, Y_train, X_test, Y_test, model)))
        if part == 'linear':
            f.write("\nw =\n{}".format(w))

    print("Written to {}!".format(args.output + '/bin_libsvm_' + part))


def binary(args: argparse.Namespace):
    print("Extracting data...")
    X_train, Y_train, X_test, Y_test = extract_data_util(
        args.train, args.test, [9, 0])
    Y_train = np.where(Y_train == 9, 1, -1)
    Y_test = np.where(Y_test == 9, 1, -1)
    print("Data extracted!")

    if args.part.find('a') != -1:
        def pred_a(w, b, X):
            return linear_prediction(w, b, X)
        cvxopt_util(args, X_train, Y_train, X_test,
                    Y_test, 'linear', linear_prod, pred_a)

    if args.part.find('b') != -1:
        def pred_b(alpha, support_Y, support_X, b, X):
            return gaussian_prediction(alpha, args.gamma, support_Y, support_X, b, X)
        cvxopt_util(args, X_train, Y_train, X_test, Y_test, 'gaussian',
                    gaussian_prod, pred_b, args.gamma)

    if args.part.find('c') != -1:
        libsvm_util(args, X_train, Y_train, X_test, Y_test, 'linear', lambda: svm_libsvm(
            X_train, Y_train, args.c, svm.LINEAR, linear_prod))
        libsvm_util(args, X_train, Y_train, X_test, Y_test, 'gaussian', lambda: svm_libsvm(
            X_train, Y_train, args.c, svm.RBF, gaussian_prod, args.gamma))


def multi(args: argparse.Namespace):
    return None


if __name__ == "__main__":
    args = get_args()

    if args.type:
        multi(args)
    else:
        binary(args)
