import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)


def RBF_kernel(a, b, lengthscale):
    """
    RBF kernel
    :param a: input vector a
    :param b: input vector b
    :param lengthscale: specify kernel parameter
    :return: kernel matrix(covariance matrix)
    """
    # loop vectorization
    sqdist = ((a[:, :, None] - b[:, :, None].T) ** 2).sum(1)
    variance = 1 # determines the average distance of function away from its mean. It's just a scale factor
    return variance * np.exp(-.5 * (1 / lengthscale) * sqdist)


def dataset_generator():
    """
    generate dataset for GP regression
    :return: true function f, training inputs X, value of training data y, test inputs X
    """
    s = 0.0005  # noise

    # Sample N input points of noisy version of the function evaluated at these points
    f = lambda x: np.sin(0.9 * x).flatten()
    X_train = np.random.uniform(-5, 5, size=(N, 1))
    y_train = f(X_train) + np.sqrt(s) * np.random.randn(N)
    # plt.plot(X_train, y_train, 'ro')

    # points we're going to make predictions at.
    X_test = np.linspace(-5, 5, n).reshape(-1, 1)
    return f, X_train, y_train, X_test


def f_prior(X_test, mu_prior, kernel_choice, kernel_parameter, num_fun):
    """
    generate GP prior
    :param X_test: arbitrary sampling points for GP prior
    :param mu_prior: mean of GP prior
    :param kernel_parameter: specify kernel parameter
    :param num_test: number of arbitrary sampling points
    :param num_fun: number of GP prior functions you want to generate
    :return: sampling function values of GP prior
    """
    num_test = len(X_test)
    if kernel_choice == 'rbf':
        kernel = RBF_kernel(X_test, X_test, kernel_parameter)  # covariance matrix for prior function
    B = np.linalg.cholesky(kernel + 1e-6 * np.eye(num_test))
    f_prior = mu_prior + np.dot(B, np.random.normal(size=(num_test, num_fun)))
    return f_prior


def prior_process(X_test, kernel_choice, kernel_parameter, num_fun):
    """
    GP prior process
    :param X_test: test data
    :param kernel_choice: choice of kernel
    :param kernel_parameter: parameter for kernel
    :param num_fun: num of prior functions that are used
    :return: GP prior function
    """
    # sampling from a multivariate Gaussian for prior function
    mu_prior = np.zeros((n, 1)) # zero mean for multivariate Gaussian
    return f_prior(X_test, mu_prior, kernel_choice, kernel_parameter, num_fun)


def prediction(X_train, X_test, y_train, kernel_choice, kernel_parameter):
    s = 0.0005  # noise variance and zero mean for noise

    if kernel_choice == 'rbf':
        K_train = RBF_kernel(X_train, X_train, kernel_parameter)
        K_s = RBF_kernel(X_train, X_test, kernel_parameter)

    L = np.linalg.cholesky(K_train + s * np.eye(N))
    m = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, m)

    # compute mean of test points for posterior
    mu_post = np.dot(K_s.T, alpha)
    K_ss = RBF_kernel(X_test, X_test, kernel_parameter)
    v = np.linalg.solve(L, K_s)

    # compute variance for test points
    var_test = np.diag(K_ss) - np.sum(v ** 2, axis=0)
    stand_devi = np.sqrt(var_test)

    # sample from test points
    L_ = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(v.T, v))
    f_post_fun = mu_post.reshape(-1, 1) + np.dot(L_, np.random.normal(size=(n, num_fun)))
    plt.axis([-5, 5, -3, 3])
    return mu_post, stand_devi, f_post_fun


def plot_prior(X_test, f_prior_fun, kernel_stand_deiv):
    """
    plot prior functions
    :param X_test: test data
    :param kernel_stand_deiv: variance that away from its mean
    :return:
    """
    plt.subplot(2, 2, 1)
    plt.fill_between(X_test.flat, 0 - kernel_stand_deiv, 0 + kernel_stand_deiv, color="#dddddd")
    plt.plot(X_test, f_prior_fun)
    plt.title('samples from the GP prior')
    plt.axis([-5, 5, -3, 3])


def plot_posterior(X_test, f_post_fun, mu_post, stand_devi):
    """
    plot posterior functions
    :param X_test: test data
    :param f_post_fun: posterior functions
    :param mu_post: mean of posterior functions
    :param stand_devi: standard derivation of posterior functions
    :return:
    """
    plt.subplot(2,2,2)
    plt.gca().fill_between(X_test.flat, mu_post - 3 * stand_devi, mu_post + 3 * stand_devi, color="#dddddd")
    plt.plot(X_test, f_post_fun)
    plt.plot(X_test, mu_post, 'r--', lw=2)
    plt.title('samples from the GP posterior')
    plt.axis([-5, 5, -3, 3])


def plot_true_diff(X_train, y_train, true_fun, mu_post, stand_devi):
    """
    plot true function and difference between true function and posterior prediction
    :param X_train: training data
    :param y_train: function value of training data
    :param true_fun: true function which get from dataset_generator function
    :param mu_post: mean of posterior functions
    :param stand_devi: standard derivation of posterior functions
    :return:
    """
    plt.subplot(2, 2, 3)
    plt.plot(X_train, y_train, 'r+', ms=20)
    plt.plot(X_test, true_fun(X_test), 'b-')
    plt.gca().fill_between(X_test.flat, mu_post - 3 * stand_devi, mu_post + 3 * stand_devi, color="#dddddd")
    plt.plot(X_test, mu_post, 'r--', lw=2)
    plt.title('Mean predictions plus 3 st.deviations')
    plt.axis([-5, 5, -3, 3])


def GP_regression(X_train, y_train, X_test, num_fun, kernel_choice, kernel_parameter):

    # GP prior
    f_prior_fun = prior_process(X_test, kernel_choice, kernel_parameter, num_fun)

    # plot prior functions
    plot_prior(X_test, f_prior_fun, kernel_stand_deiv)

    # make prediction for test data by posterior
    mu_post, stand_devi, f_post_fun = prediction(X_train, X_test, y_train, kernel_choice, kernel_parameter)

    # plot posterior functions
    plot_posterior(X_test, f_post_fun, mu_post, stand_devi)

    # plot true function and difference between true function and posterior prediction
    plot_true_diff(X_train, y_train, true_fun, mu_post, stand_devi)

    plt.show()


if __name__ == "__main__":

    N = 5           # number of training points
    n = 100          # number of test points

    # hyper-parameters
    num_fun = 10    # number of prior function
    kernel_parameter = 1
    kernel_stand_deiv = 1 # standard deviation for kernel
    kernel_choice = 'rbf' # can be 'rbf', 'per', 'lin'

    # generate dataset for GP regression
    true_fun, X_train, y_train, X_test = dataset_generator()

    # GP regression
    GP_regression(X_train, y_train, X_test, num_fun, kernel_choice, kernel_parameter)