from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
np.set_printoptions(precision=3, suppress=True)


def RBF_kernel(a, b, sigma, l):
    """
    RBF kernel
    :param a: input vector a
    :param b: input vector b
    :param l: lengthscale
    :return: kernel matrix(covariance matrix)
    """
    #sigma = 1  # determines the average distance of function away from its mean. It's just a scale factor
    # loop vectorization
    sqdist = ((a[:, :, None] - b[:, :, None].T) ** 2).sum(1)
    return (sigma ** 2) * np.exp(-.5 * (1 / (l ** 2)) * sqdist)


def lin_kernel(a, b, c):
    """
    linear kernel
    :param a: input vector a
    :param b: input vector b
    :param c: offset c determines the x-coordinate of the point that all the lines in the posterior go though
    :return: kernel matrix(covariance matrix)
    """
    output_variance = 1
    fun_mean = 0 # zero-mean Gaussian priors
    dot_product = np.dot(a - c, b.T - c)
    return fun_mean + output_variance * dot_product


def per_kernel(a, b, parameters):
    """
    periodic kernel
    :param a: input vector a
    :param b: input vector b
    :param parameters: period and lengthscale
    :return: kernel matrix(covariance matrix)
    """
    output_variance = 1
    p, l = parameters
    num_a = len(a)
    num_b = len(b)
    l2_norm = np.absolute(np.tile(a, (1, num_b)) - np.tile(b.T, (num_a, 1)))
    f = lambda x: output_variance * np.exp(-2 * (np.sin(np.pi * l2_norm / p))**2 / l**2)
    return f(l2_norm)


def dataset_generator(N, n):
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
    s = 0.0005  # noise variance and zero mean for noise
    sigma = 1
    num_test = len(X_test)
    if kernel_choice == 'rbf':
        kernel = RBF_kernel(X_test, X_test, sigma, kernel_parameter)  # covariance matrix for prior function
    if kernel_choice == 'lin':
        kernel = lin_kernel(X_test, X_test, kernel_parameter)
    if kernel_choice == 'per':
        kernel = per_kernel(X_test, X_test, kernel_parameter)
    B = np.linalg.cholesky(kernel + s * np.eye(num_test))
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


def prediction(X_train, X_test, y_train, kernel_choice, l):
    s = 0.0005  # noise variance and zero mean for noise
    sigma = 1

    if kernel_choice == 'rbf':
        K_train = RBF_kernel(X_train, X_train, sigma, l)
        K_s = RBF_kernel(X_train, X_test, sigma, l)
        K_ss = RBF_kernel(X_test, X_test, sigma, l)
    if kernel_choice == 'lin':
        K_train = lin_kernel(X_train, X_train, l)
        K_s = lin_kernel(X_train, X_test, l)
        K_ss = lin_kernel(X_test, X_test, l)
    if kernel_choice == 'per':
        K_train = per_kernel(X_train, X_train, l)
        K_s = per_kernel(X_train, X_test, l)
        K_ss = per_kernel(X_test, X_test, l)

    L = np.linalg.cholesky(K_train + s * np.eye(N))
    m = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, m)

    # compute mean of test points for posterior
    mu_post = np.dot(K_s.T, alpha)
    v = np.linalg.solve(L, K_s)

    # compute variance for test points
    var_test = np.diag(K_ss) - np.sum(v ** 2, axis=0)
    stand_devi = np.sqrt(var_test)

    # compute log marginal likelihood
    log_marg_likelihood = -.5 * np.dot(y_train.T, alpha) - np.diagonal(L).sum(0) - n/2 * np.log(2*np.pi)

    # sample from test points
    L_ = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(v.T, v))
    f_post_fun = mu_post.reshape(-1, 1) + np.dot(L_, np.random.normal(size=(n, num_fun)))
    return mu_post, stand_devi, f_post_fun


def plot_rbf_kernel():
    """
    plot RBF kernel
    :return:
    """
    plt.subplot(4, 2, 1)
    mu = 3
    variance = 1
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 3 * variance, mu + 3 * variance, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma))
    plt.title('RBF kernel')


def plot_lin_kernel():
    """
    plot linear kernel
    :return:
    """
    # fix x_ = 3
    x_ = 3
    output_variance = 1
    c = 0 # offset
    mean_function = 0
    x = np.linspace(x_ - 3 * output_variance, x_ + 3 * output_variance, 100)
    f = lambda x: mean_function + output_variance * (x-c) * (x_-c)
    plt.subplot(4, 2, 1)
    plt.plot(x, f(x))
    plt.title('linear kernel')


def plot_per_kernel():
    output_variance = 1
    p, l = np.array([1, 1])
    x_ = 3
    x = np.linspace(x_ - 3 * output_variance, x_ + 3 * output_variance, 100)
    f = lambda x: output_variance * np.exp(-2 * (np.sin(np.pi * x / p)) ** 2 / l ** 2)
    plt.subplot(4, 2, 1)
    plt.plot(x, f(x))
    plt.title('periodic kernel')

def plot_kernel(kernel_choice):
    """
    plot different kernel based on choice of kernel
    :param kernel_choice: 'rbf','lin','per'
    :return:
    """
    if kernel_choice == 'rbf':
        plot_rbf_kernel()
    if kernel_choice == 'lin':
        plot_lin_kernel()
    if kernel_choice == 'per':
        plot_per_kernel()


def plot_prior(X_test, f_prior_fun, kernel_stand_deiv):
    """
    plot prior functions
    :param X_test: test data
    :param kernel_stand_deiv: variance that away from its mean
    :return:
    """
    plt.subplot(4, 2, 3)
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
    plt.subplot(4, 2, 5)
    plt.gca().fill_between(X_test.flat, mu_post - 3 * stand_devi, mu_post + 3 * stand_devi, color="#dddddd")
    plt.plot(X_test, f_post_fun)
    plt.plot(X_test, mu_post, 'r--', lw=2)
    plt.title('samples from the GP posterior')
    plt.axis([-5, 5, -3, 3])


def plot_true_diff(X_train, X_test, y_train, true_fun, mu_post, stand_devi):
    """
    plot true function and difference between true function and posterior prediction
    :param X_train: training data
    :param y_train: function value of training data
    :param true_fun: true function which get from dataset_generator function
    :param mu_post: mean of posterior functions
    :param stand_devi: standard derivation of posterior functions
    :return:
    """
    plt.subplot(4, 2, 7)
    plt.plot(X_train, y_train, 'r+', ms=20)
    plt.plot(X_test, true_fun(X_test), 'b-')
    plt.gca().fill_between(X_test.flat, mu_post - 3 * stand_devi, mu_post + 3 * stand_devi, color="#dddddd")
    plt.plot(X_test, mu_post, 'r--', lw=2)
    plt.title('Mean predictions plus 3 st.deviations')
    plt.axis([-5, 5, -3, 3])


def GP_regression(X_train, y_train, X_test, num_fun, kernel_choice, kernel_parameter):
    # plot kernel function
    plot_kernel(kernel_choice)

    # GP prior
    f_prior_fun = prior_process(X_test, kernel_choice, kernel_parameter, num_fun)

    # plot prior functions
    plot_prior(X_test, f_prior_fun, kernel_stand_deiv)

    # make prediction for test data by posterior
    mu_post, stand_devi, f_post_fun = prediction(X_train, X_test, y_train, kernel_choice, kernel_parameter)

    # plot posterior functions
    plot_posterior(X_test, f_post_fun, mu_post, stand_devi)

    # plot true function and difference between true function and posterior prediction
    plot_true_diff(X_train, X_test, y_train, true_fun, mu_post, stand_devi)

    plt.show()


if __name__ == "__main__":

    N = 5   # number of training points
    n = 100 # number of test points

    # hyper-parameters
    num_fun = 10          # number of prior function
    #kernel_parameter = [1, 1]  # parameter for kernel per
    kernel_parameter = 1
    kernel_stand_deiv = 1 # standard deviation for kernel
    kernel_choice = 'rbf' # can be 'rbf', 'per', 'lin'

    # generate dataset for GP regression
    true_fun, X_train, y_train, X_test = dataset_generator(N, n)

    # GP regression
    GP_regression(X_train, y_train, X_test, num_fun, kernel_choice, kernel_parameter)