from q2_util import *
import argparse
from os import makedirs
from time import time
import pickle


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
    parser.add_argument('-mc', dest='model_cvxopt', type=str,
                        help="extract model from a pickle dump for cvxopt")
    parser.add_argument('-ms', dest='model_libsvm', type=str,
                        help="extract model from a pickle dump for libsvm")
    parser.add_argument('-pc', dest='pred_cvxopt', type=str,
                        help="extract prediction from a pickle dump for cvxopt")
    parser.add_argument('-ps', dest='pred_libsvm', type=str,
                        help="extract prediction from a pickle dump for libsvm")
    parser.add_argument('-k', dest='k', type=int, default=10,
                        help="number of classes (for multi class)")

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
    X_train, Y_train, X_test, Y_test = extract_data_util(
        args.train, args.test, [9, 0])
    Y_train = np.where(Y_train == 9, 1, -1)
    Y_test = np.where(Y_test == 9, 1, -1)

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
    X_train, Y_train, X_test, Y_test = extract_data_util(
        args.train, args.test, range(args.k))

    if args.part.find('a') != -1 or args.part.find('c') != -1:
        if args.model_cvxopt is not None:
            print("Unpickling model...")
            classifier_cvxopt, train_t = pickle.load(
                open(args.model_cvxopt, 'rb'))
            print("Model unpickled!")
        else:
            print("Training model...")
            train_t = time()

            classifier_cvxopt = kC2_classifier_cvxopt(
                X_train, Y_train, args.c, args.tol, args.gamma, args.k)

            train_t = time() - train_t
            print("Model trained!\nPickling model...")

            pickle.dump((classifier_cvxopt, train_t), open(
                args.output + '/multi_model_cvxopt', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            print("Model pickled to {}!".format(
                args.output + '/multi_model_cvxopt'))

        if args.pred_cvxopt is not None:
            print("Unpickling prediction...")
            prediction_cvxopt, pred_t = pickle.load(
                open(args.pred_cvxopt, 'rb'))
            print("Prediction unpickled!")
        else:
            print("Making prediction...")
            pred_t = time()

            prediction_cvxopt = predict_k_cvxopt(
                classifier_cvxopt, X_test, args.gamma, args.k)

            pred_t = time() - pred_t
            print("Predictions made!\nPickling model...")

            pickle.dump((prediction_cvxopt, pred_t), open(
                args.output + '/multi_pred_cvxopt', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            print("Model pickled to {}!".format(
                args.output + '/multi_pred_cvxopt'))

    if args.part.find('a') != -1:
        with open(args.output + '/multi_cvxopt', 'w+') as f:
            print("Computing accuracy and writing to file...")

            f.write("training time         = {}\n".format(train_t))
            f.write("prediction time       = {}\n".format(pred_t))
            f.write("accuracy              = {}\n".format(
                accuracy(prediction_cvxopt, Y_test)))

            print("Written to {}!".format(args.output + '/multi_cvxopt'))

    if args.part.find('b') != -1 or args.part.find('c') != -1:
        if args.model_libsvm is not None:
            print("Loading model...")
            classifier_libsvm = svm.svm_load_model(args.model_libsvm)
            with open(args.model_libsvm + '_t') as f:
                train_t = int(f.readline())
            print("Model loaded!")
        else:
            print("Training model...")
            train_t = time()

            classifier_libsvm = kC2_classifier_libsvm(
                X_train, Y_train, args.c, svm.RBF, args.gamma)

            train_t = time() - train_t
            print("Model trained!\nSaving model...")

            svm.svm_save_model(
                args.output + '/multi_model_libsvm', classifier_libsvm)
            with open(args.output + '/multi_model_libsvm_t', 'w+') as f:
                f.write(str(train_t))

            print("Model saved to {}!".format(
                args.output + '/multi_model_libsvm(_t)'))

        if args.pred_libsvm is not None:
            print("Unpickling prediction...")
            prediction_libsvm, pred_t = pickle.load(
                open(args.pred_libsvm, 'rb'))
            print("Prediction unpickled!")
        else:
            print("Making prediction...")
            pred_t = time()

            prediction_libsvm = np.array(
                [svm.svm_predict(Y_test.T[0], X_test, classifier_libsvm, '-q')[0]], np.int32).T

            pred_t = time() - pred_t
            print("Predictions made!\nPickling model...")

            pickle.dump((prediction_libsvm, pred_t), open(
                args.output + '/multi_pred_libsvm', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            print("Model pickled to {}!".format(
                args.output + '/multi_pred_libsvm'))

    if args.part.find('b') != -1:
        with open(args.output + '/multi_libsvm', 'w+') as f:
            print("Computing accuracy and writing to file...")

            f.write("training time   = {}\n".format(train_t))
            f.write("prediction time = {}\n".format(pred_t))
            f.write("accuracy        = {}".format(
                accuracy(prediction_libsvm, Y_test)))

        print("Written to {}!".format(args.output + '/multi_libsvm'))


if __name__ == "__main__":
    args = get_args()

    if args.type:
        multi(args)
    else:
        binary(args)
