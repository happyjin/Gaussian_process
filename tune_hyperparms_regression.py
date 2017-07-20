from __future__ import division
from GP_regression import dataset_generator, RBF_kernel, plot_posterior, plot_true_diff
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)


def gradient_ascent(a, b, sigma, l, alpha, K_y):
    """
    tune hyperparameters sigma and l for RBF kernel
    :param a: input vector a
    :param b: input vector b
    :param sigma: output variance determines the average distance of your function away from its mean
    :param l: lengthscale determines the length of the 'wiggles' in your function.
    :param alpha: equals to K_inv * y
    :param K_y: K_inv
    :return: current sigmal and l
    """
    step_size = 0.01
    sqdist = ((a[:, :, None] - b[:, :, None].T) ** 2).sum(1)

    # fix the output variance of RBF kernel in order to visualize it in one dimension
    '''
    # tune hyperparameter sigma
    sigma_grad = 2 * sigma * np.exp(-.5*sqdist/(l**2))
    sigma_matrix = np.dot(np.dot(alpha, alpha.T) - K_y, sigma_grad)
    tr_sigma = np.diagonal(sigma_matrix).sum()
    sigma_var = .5 * tr_sigma
    '''
    # tune hyperparameter l
    l_grad = sigma**2 * np.exp(-.5*sqdist/(l**2)) * (sqdist/l**3)
    l_matrix = np.dot(np.dot(alpha, alpha.T) - K_y, l_grad)
    tr_l = np.diagonal(l_matrix).sum()
    l_var = .5 * tr_l

    # gradient ascent to maximum log marginal likelihood simultaneously
    '''
    sigma = sigma + step_size * sigma_var
    '''
    l = l + step_size * l_var
    return sigma, l


def bayesian_opt(X, y):
    s = 0.0005  # noise variance and zero mean for noise



def tune_hyperparms_first(X_train, X_test, y_train, num_fun, sigma, l):
    """
    maximize log marginal likelihood using gradient ascent
    :param X_train: training data
    :param X_test: test data
    :param y_train: training target
    :param num_fun: number of functions
    :param sigma: the output variance of RBF kernel
    :param l: lengthscalar
    :return: mean, standard derivation, posterior function
    """
    s = 0.0005  # noise variance and zero mean for noise
    log_marg_likelihood_old = 0
    tolerance = 0.001

    for i in range(10000):
        # choose RBF kernel in this regression case
        K_train = RBF_kernel(X_train, X_train, sigma, l)
        K_s = RBF_kernel(X_train, X_test, sigma, l)
        K_ss = RBF_kernel(X_test, X_test, sigma, l)

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
        log_marg_likelihood = -.5 * np.dot(y_train.T, alpha) - np.diagonal(L).sum(0) - n / 2 * np.log(2 * np.pi)

        # tune the hyperparameters for RBF kernel
        K_y_inv = np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
        sigma, l = gradient_ascent(X_train, X_train, sigma, l, alpha.reshape(-1, 1), K_y_inv)

        error = np.sqrt(np.sum((log_marg_likelihood - log_marg_likelihood_old) ** 2))
        log_marg_likelihood_old = log_marg_likelihood
        if error <= tolerance:
            print "The hyperparameter tuning function has already converged after " + `i + 1` + " iterations!"
            print "The error is " + `error`
            print "training end!"
            break

    print sigma, l
    # sample from test points, in other words, make prediction
    L_ = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(v.T, v))
    f_post_fun = mu_post.reshape(-1, 1) + np.dot(L_, np.random.normal(size=(n, num_fun)))
    plt.axis([-5, 5, -3, 3])
    return mu_post, stand_devi, f_post_fun


def tune_hyperparms_second(X_train, X_test, y_train, num_fun, sigma, l):
    s = 0.0005  # noise variance and zero mean for noise

    for i in range(len(l)):
        log_marg_likelihood = np.zeros(len(l))
        # choose RBF kernel in this regression case
        K_train = RBF_kernel(X_train, X_train, sigma, l[i])
        K_s = RBF_kernel(X_train, X_test, sigma, l[i])
        K_ss = RBF_kernel(X_test, X_test, sigma, l[i])

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
        log_marg_likelihood[i] = -.5 * np.dot(y_train.T, alpha) - np.diagonal(L).sum(0) - n / 2 * np.log(2 * np.pi)

        # Bayesian optimization
        bayesian_opt(l, log_marg_likelihood)

def tune_hyperparms_gradient(X_train, X_test, y_train, num_fun):
    """
    tune hyperparameters by maximizing the log marginal likelihood
    :param X_train: training data
    :param X_test: test data
    :param y_train: training targets
    :param num_fun: number of functions
    :return:
    """
    sigma = 1 # fix the output variance of RBF kernel
    l = 5  # initial hyperparm
    # tune hyperparameters of RBF kernel in the regression using gradient ascent
    mu_post, stand_devi, f_post_fun = tune_hyperparms_first(X_train, X_test, y_train, num_fun, sigma, l)
    # plot posterior functions
    plot_posterior(X_test, f_post_fun, mu_post, stand_devi)
    # plot true function and difference between true function and posterior prediction
    plot_true_diff(X_train, X_test, y_train, true_fun, mu_post, stand_devi)


def tune_hyperparms_BO(X_train, X_test, y_train, num_fun):
    sigma = 1 # fix the output variance of RBF kernel
    # random pick up two initial hyperparm
    l = np.array([1.7, 3.3])#np.random.uniform(0.1,5,2)
    # tune hyperparameters of RBF kernel in regression using Bayesian optimization
    tune_hyperparms_second(X_train, X_test, y_train, num_fun, sigma, l)


if __name__ == "__main__":
    N = 3  # number of training points
    n = 100  # number of test points
    sigma = 1 # fix the output variance of RBF kernel
    num_fun = 10  # number of prior function

    # generate dataset for GP regression
    true_fun, X_train, y_train, X_test = dataset_generator(N, n)

    # using gradient ascent to maximize log marginal likelihood in order to tune the hyperparameters
    #tune_hyperparms_gradient(X_train, X_test, y_train, num_fun)

    # using Bayesian optimization in order to tune the hyperparameters
    tune_hyperparms_BO(X_train, X_test, y_train, num_fun)

    plt.show()