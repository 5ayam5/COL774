import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from os import makedirs

parser = argparse.ArgumentParser(description="Program to solve Q1 \
                                  of COL774 Assignment 1")
parser.add_argument('train', type=str,
                    help="train the model present in directory 'train' as \
                    linearX.csv and linearY.csv")
parser.add_argument('-o', dest='output', type=str, default='./output',
                    help="output directory (if directory does not exist,\
                    it is created) [default './output/']")
parser.add_argument('-eta', type=float, default=0.05,
                    help="learning rate [default 0.05]")
parser.add_argument('-eps', type=float, default=1e-15,
                    help="convergence criteria [default 1e-15]")
parser.add_argument('-d', dest='display', type=str, default='',
                    help="display the plots and animations")
parser.add_argument('-f', dest='figure', type=str, default='',
                    help="save the figure")
args = parser.parse_args()


# read data from csv file
def extract_data(v: str):
    try:
        # return column vector of the csv file
        return np.loadtxt(open(args.train + 'linear' + v + '.csv'), ndmin=2)
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
    j_prev, j = cost(X, Y, Theta), 0
    Thetas = [(Theta[0][0], Theta[1][0], j_prev)]
    i = 0
    while abs(j - j_prev) > epsilon:
        j_prev = j
        Theta -= eta * grad(X, Y, Theta)
        j = cost(X, Y, Theta)
        Thetas.append((Theta[0][0], Theta[1][0], j))
        i += 1
    return Theta, i, np.array(Thetas).T


# generate the mesh data for plotting
def generate_mesh(X: np.ndarray, Theta: np.ndarray):
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    theta0, theta1 = np.meshgrid(
        *np.linspace(-Theta.T[0], 2 * Theta.T[0], 100).T)
    J = np.apply_along_axis(
        lambda theta: cost_function(X, Y, np.reshape(theta, (-1, 1))),
        2, np.stack([theta0, theta1], axis=-1))
    return theta0, theta1, J


if __name__ == '__main__':
    try:
        makedirs(args.output, exist_ok=True)
    except Exception as e:
        print("error: " + str(e))
        exit(2)

    #################
    # extracting data
    X, Y = extract_data('X'), extract_data('Y')
    if X is None or Y is None:
        print("error: could not load data")
        exit(1)
    X = normalise(X)

    #################
    # code for part a
    eta, epsilon = args.eta, args.eps
    Theta, iterations, Thetas = gradient_descent(X, Y, eta, epsilon)
    with open(args.output + "/a", "w+") as out:
        out.writelines("\n".join(["learning rate = " + str(eta),
                                  "epsilon       = " + str(epsilon),
                                  "theta         = [" + str(Theta[0][0]) +
                                  ", " + str(Theta[1][0]) + "]",
                                  "#iterations   = " + str(iterations)]))

    #################
    # code for part b
    fig_b, axes_b = plt.subplots()
    axes_b.set_title("Linear Regression")
    axes_b.set_xlabel("Acidity")
    axes_b.set_ylabel("Density")
    axes_b.scatter(X, Y, label='training data')
    axes_b.plot(X, X * Theta[1] + Theta[0], color='orange',
                label='trained model')
    axes_b.legend()
    fig_b.tight_layout()
    if args.display.find('b') != -1:
        plt.show()
    if args.figure.find('b') != -1:
        fig_b.savefig(args.output + "/b.png")
    plt.close()

    #################
    # code for part c
    theta0, theta1, J = generate_mesh(X, Theta)
    fig_c = plt.figure()
    axes_c = fig_c.add_subplot(title='3D Mesh', projection='3d',
                               xlabel='$theta_0$', ylabel='$theta_1$',
                               zlabel='cost function')
    surf = axes_c.plot_surface(theta0, theta1, J, color='grey',
                               label='cost mesh')
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    learn, = axes_c.plot([], [], [], color='red', label='learning theta')
    axes_c.legend()
    fig_c.tight_layout()

    def update(iteration):
        learn.set_data(Thetas[:2, :iteration+1])
        learn.set_3d_properties(Thetas[2, :iteration+1])
        return learn,
    anim = animation.FuncAnimation(fig_c, update,
                                   frames=range(Thetas.shape[1]),
                                   interval=200, blit=True)
    if args.display.find('c') != -1:
        plt.show()
    update(iterations)
    if args.figure.find('c') != -1:
        anim.save(args.output + '/c.mkv', writer='imagemagick', fps=10)
        update(iterations)
        fig_c.savefig(args.output + "/c.png")
    plt.close()

    #################
    # code for part d
    fig_d, axes_d = plt.subplots()
    axes_d.set_title('Contours for eta = ' + str(eta))
    axes_d.set_xlabel('$theta_0$')
    axes_d.set_ylabel('$theta_1$')
    axes_d.contour(theta0, theta1, J, 100)
    learn, = axes_d.plot([], [], color='red', marker='x',
                         linestyle='None', label='learning theta')
    axes_d.legend()
    fig_d.tight_layout()

    def update(iteration):
        learn.set_data(Thetas[:2, :iteration+1])
        return learn,
    anim = animation.FuncAnimation(fig_d, update,
                                   frames=range(Thetas.shape[1]),
                                   interval=200, blit=True)
    if args.display.find('d') != -1:
        plt.show()
    if args.figure.find('d') != -1:
        anim.save(args.output + '/d.mkv', writer='imagemagick', fps=10)
        update(iterations)
        fig_d.savefig(args.output + "/d.png")
    plt.close()

    #################
    # code for part e
    fig_e, axes_e = plt.subplots(3)
    learn = [None for _ in range(3)]
    Thetass = [None for _ in range(3)]
    for i, eta in enumerate([0.001, 0.025, 0.1]):
        Theta, iterations, Thetass[i] = gradient_descent(X, Y, eta, epsilon)
        axes_e[i].set_title('Contours for eta = ' + str(eta))
        axes_e[i].set_xlabel('$theta_0$')
        axes_e[i].set_ylabel('$theta_1$')
        axes_e[i].contour(theta0, theta1, J, 100)
        learn[i], = axes_e[i].plot([], [], color='red', linestyle='None',
                                   marker='x', label='learning theta')
        axes_e[i].legend()
    fig_e.tight_layout()

    def update(iteration):
        for i, Thetas in enumerate(Thetass):
            learn[i].set_data(Thetas[:2, :iteration+1])
        return learn
    anim = animation.FuncAnimation(fig_d, update,
                                   frames=range(Thetass[0].shape[1]),
                                   interval=200, blit=True)
    if args.display.find('e') != -1:
        plt.show()
    if args.figure.find('e') != -1:
        update(Thetass[0].shape[1])
        fig_e.savefig(args.output + "/e.png")
    plt.close()
