from functools import reduce
import numpy as np
import cvxopt
import libsvm.svmutil as svm


def extract_data(file: str, subset: list[int]):
    data = np.genfromtxt(file, delimiter=',')
    X = data[:, :-1] / 255
    Y = np.array(data[:, -1], np.int32)

    indices = np.where(np.vectorize(lambda y: y in subset)(Y))
    X = X[indices]
    Y = Y[indices]
    Y = np.reshape(Y, (Y.shape[0], 1))

    return X, Y


def extract_data_util(train: str, test: str, subset: list[int]):
    print("Extracting data...")
    X_train, Y_train = extract_data(train, subset)
    X_test, Y_test = extract_data(test, subset)
    print("Data extracted!")
    return X_train, Y_train, X_test, Y_test


def linear_prod(X1: np.ndarray, X2: np.ndarray, *args):
    return np.matmul(X1, X2.T)


def gaussian_prod(X1: np.ndarray, X2: np.ndarray, gamma: float):
    prod1 = np.reshape(np.einsum('ij,ij->i', X1, X1), (X1.shape[0], 1))
    prod2 = np.reshape(np.einsum('ij,ij->i', X2, X2), (X2.shape[0], 1))
    prod = prod1 + prod2.T - 2 * np.matmul(X1, X2.T)
    return np.exp(-gamma * prod)


def svm_cvxopt(X: np.ndarray, Y: np.ndarray, c: float, tol: float, find_prod, gamma: float = None):
    shape = Y.shape

    prod = find_prod(X, X, gamma)
    P = cvxopt.matrix(np.matmul(Y, Y.T) * prod)

    q = cvxopt.matrix(-np.ones(shape))

    G = np.identity(shape[0])
    G = cvxopt.matrix(np.append(G, -G, axis=0))

    h = np.ones(shape)
    h = cvxopt.matrix(np.append(c * h, 0 * h, axis=0))

    A = cvxopt.matrix(Y.T, tc='d')

    b = cvxopt.matrix(0.0)

    sol = cvxopt.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    if sol['status'] == "unknown":
        return None

    alpha = np.reshape(np.array(sol['x']), shape)
    indices = [i for i in range(shape[0]) if alpha[i] > tol]
    alpha = alpha[indices]
    X, Y = X[indices], Y[indices]
    inner_prod = np.sum(alpha * Y * find_prod(X, X, gamma), 0)

    M = max(range(len(indices)), key=lambda i: -float("inf")
            if Y[i] == 1 or alpha[i] >= c - tol else inner_prod[i])
    m = min(range(len(indices)), key=lambda i: float("inf")
            if Y[i] == -1 or alpha[i] >= c - tol else inner_prod[i])
    b = -(inner_prod[M] + inner_prod[m]) / 2

    return indices, alpha, b


def linear_prediction(w: np.ndarray, b: float, X: np.ndarray):
    return np.where(linear_prod(X, w.T) + b >= 0, 1, -1)


def gaussian_prediction(alpha: np.ndarray, gamma: float, support_Y: np.ndarray, support_X: np.ndarray, b: float, X: np.ndarray):
    return np.reshape(np.where(np.sum(alpha * support_Y * gaussian_prod(support_X, X, gamma), 0) + b >= 0, 1, -1), (X.shape[0], 1))


def accuracy(pred_Y: np.ndarray, Y: np.ndarray):
    return 100 * sum(pred_Y == Y)[0] / Y.shape[0]


def accuracy_util_cvxopt(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, pred):
    train_accuracy = accuracy(pred(X_train), Y_train)
    test_accuracy = accuracy(pred(X_test), Y_test)
    return train_accuracy, test_accuracy


def svm_libsvm(X: np.ndarray, Y: np.ndarray, c: float, kernel, prod, gamma: float = None):
    params = '-t {} -c {} -q'.format(kernel, c)
    if gamma:
        params += ' -g {}'.format(gamma)

    model = svm.svm_train(Y.T[0], X, params)
    indices = model.get_sv_indices()
    for i in range(len(indices)):
        indices[i] -= 1
    alpha = np.abs(np.array(model.get_sv_coef(), ndmin=2))

    X, Y = X[indices], Y[indices]
    inner_prod = np.sum(alpha * Y * prod(X, X, gamma), 0)
    M = max(range(len(alpha)), key=lambda i: -
            float("inf") if Y[i] == 1 else inner_prod[i])
    m = min(range(len(alpha)), key=lambda i: float(
        "inf") if Y[i] == -1 else inner_prod[i])
    b = -(inner_prod[M] + inner_prod[m]) / 2

    return indices, alpha, model, b


def accuracy_util_libsvm(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, model):
    train_accuracy = svm.svm_predict(Y_train.T[0], X_train, model, '-q')[1][0]
    test_accuracy = svm.svm_predict(Y_test.T[0], X_test, model, '-q')[1][0]
    return train_accuracy, test_accuracy


class Classifier():
    X_sv: np.ndarray
    Y_sv: np.ndarray
    indices: list[int]
    alpha: np.ndarray
    b: float


def kC2_classifier_cvxopt(X: np.ndarray, Y: np.ndarray, c: float, tol: float, gamma: float, k: int):
    classes = [list(np.where(Y == i)[0]) for i in range(k)]
    classifier = dict()

    for i in range(k):
        for j in range(i + 1, k):
            X_ij, Y_ij = X[classes[i] + classes[j]], Y[classes[i] + classes[j]]
            Y_ij = np.where(Y_ij == j, 1, -1)
            c_ij = classifier[i, j] = Classifier()

            c_ij.indices, c_ij.alpha, c_ij.b = svm_cvxopt(
                X_ij, Y_ij, c, tol, gaussian_prod, gamma)
            c_ij.X_sv, c_ij.Y_sv = X_ij[c_ij.indices], Y_ij[c_ij.indices]

            print("Classifier trained for classes {} and {}!".format(i, j))

    return classifier


def predict_k_cvxopt(classifier: dict[tuple[int, int], Classifier], X: np.ndarray, gamma: float, k: int):
    votes = np.zeros((X.shape[0], k), np.int32)
    winner = [dict() for _ in range(X.shape[0])]

    for i in range(k):
        for j in range(i + 1, k):
            c_ij = classifier[i, j]
            preds = gaussian_prediction(
                c_ij.alpha, gamma, c_ij.Y_sv, c_ij.X_sv, c_ij.b, X)
            for m, (example, pred) in enumerate(zip(votes, preds)):
                if pred == 1:
                    example[j] += 1
                    winner[m][i, j] = j
                else:
                    example[i] += 1
                    winner[m][i, j] = i

    best = map(lambda row: np.where(row == row.max())[0], votes)
    pred = np.empty((X.shape[0], 1), np.int32)
    for m, row in enumerate(best):
        pred[m] = reduce(lambda i, j: winner[m][i, j], row)

    return pred


def kC2_classifier_libsvm(X: np.ndarray, Y: np.ndarray, c: float, kernel, gamma: float):
    params = '-t {} -c {} -q'.format(kernel, c)
    if gamma:
        params += ' -g {}'.format(gamma)
    return svm.svm_train(Y.T[0], X, params)


def confusion_matrix(Y: np.ndarray, pred_Y: np.ndarray, k: int):
    confusion = np.zeros((k, k), np.int32)
    for y, pred_y in zip(Y.T[0], pred_Y.T[0]):
        confusion[y][pred_y] += 1
    return confusion


def misclassified(Y: np.ndarray, pred_Y: np.ndarray):
    ret = []
    for i, (y, y_pred) in enumerate(zip(Y, pred_Y)):
        if y != y_pred:
            ret.append((i, y[0], y_pred[0]))
    return ret


def kC2_cross_classifier(X: np.ndarray, Y: np.ndarray, c: float, kernel, gamma: float, fold: int):
    m = Y.shape[0]
    m //= fold
    X_test, X_train = np.split(X, [m])
    Y_test, Y_train = np.split(Y, [m])

    cross_accuracy = 0
    for i in range(fold):
        model = kC2_classifier_libsvm(X_train, Y_train, c, kernel, gamma)
        cross_accuracy += accuracy(np.array(
            [svm.svm_predict(Y_test.T[0], X_test, model, '-q')[0]], np.int32).T, Y_test)
        if i < fold - 1:
            X_train[i * m:(i + 1) * m], X_test = X_test, X_train[i * m:(i + 1) * m].copy()
        print("Fold {} done!".format(i))

    return cross_accuracy / fold
