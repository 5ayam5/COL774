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
    X_train, Y_train = extract_data(train, subset)
    X_test, Y_test = extract_data(test, subset)
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


def accuracy(X: np.ndarray, Y: np.ndarray, pred):
    return 100 * sum(pred(X) == Y)[0] / Y.shape[0]


def accuracy_util_cvxopt(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, pred):
    train_accuracy = accuracy(X_train, Y_train, pred)
    test_accuracy = accuracy(X_test, Y_test, pred)
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
