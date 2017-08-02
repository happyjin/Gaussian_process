from __future__ import division
from GP_regression import dataset_generator, RBF_kernel, plot_posterior, plot_true_diff, prediction
from scipy.stats import norm
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)


def plot_BO(X_train, y_train, X_test, f_post_fun, mu_post, stand_devi):
    """
    plot Bayesian optimization figure in order to visualize it
    :param X_train: training data
    :param y_train: training targets
    :param X_test: test data
    :param f_post_fun: GP posterior function
    :param mu_post: mean of GP posterior function
    :param stand_devi: standard deviation
    :return:
    """
    plt.subplot(2, 1, 1)
    plt.plot(X_train, y_train, 'r+', ms=20)
    plt.gca().fill_between(X_test.flat, mu_post - 3 * stand_devi, mu_post + 3 * stand_devi, color="#dddddd")
    #plt.plot(X_test, f_post_fun)
    #plt.plot(X_test, mu_post, 'r--', lw=2)
    plt.plot(X_test, mu_post, linewidth=1)
    plt.title('Bayesian Optimization')
    plt.show()


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


def bayesian_opt(X_train, X_test, y_train):
    """
    compute current GP for Bayesian optimization
    :param X_train: training data
    :param X_test: test data
    :param y_train: training targets
    :return: mean of GP posterior function, standard deviation, GP posterior function
    """
    s = 0.0001  # noise variance and zero mean for noise
    n = len(X_test) # number of test points
    N = len(X_train) # number of training points
    num_fun = 1
    #print X_test

    K = RBF_kernel(X_train, X_train, 1, 1)
    K_s = RBF_kernel(X_train, X_test, 1, 1)
    K_ss = RBF_kernel(X_test, X_test, 1, 1)

    L = np.linalg.cholesky(K + s * np.eye(N))
    m = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, m)

    # compute mean of test points for posterior
    mu_post = np.dot(K_s.T, alpha)
    v = np.linalg.solve(L, K_s)

    # compute variance for test points
    var_test = np.diag(K_ss) - np.sum(v ** 2, axis=0)
    stand_devi = np.sqrt(var_test)

    # sample from test points, in other words, make prediction
    L_ = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(v.T, v))
    f_post_fun = mu_post.reshape(-1, 1) + np.dot(L_, np.random.normal(size=(n, num_fun)))
    #plot_BO(X_train, y_train, X_test, f_post_fun, mu_post, stand_devi)
    return mu_post, stand_devi, f_post_fun


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
    n = len(X_train)
    N = len(X_test)

    for i in range(10000):
        # choose RBF kernel in this regression case
        K_train = RBF_kernel(X_train, X_train, sigma, l)
        K_s = RBF_kernel(X_train, X_test, sigma, l)
        K_ss = RBF_kernel(X_test, X_test, sigma, l)

        L = np.linalg.cholesky(K_train + s * np.eye(n))
        m = np.linalg.solve(L, y_train)
        alpha = np.linalg.solve(L.T, m)

        # compute mean of test points for posterior
        mu_post = np.dot(K_s.T, alpha)
        v = np.linalg.solve(L, K_s)

        # compute variance for test points
        var_test = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        stand_devi = np.sqrt(var_test)

        # compute log marginal likelihood
        #log_marg_likelihood = -.5 * np.dot(y_train.T, alpha) - np.diagonal(L).sum(0) - n / 2 * np.log(2 * np.pi)
        log_marg_likelihood = -.5 * np.dot(y_train.T, alpha) - np.log(np.diagonal(L)).sum(0) - n / 2.0 * np.log(2 * np.pi)

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

    optimal_likelihood = log_marg_likelihood
    print 'optimal lenghscalar is: ' + `l[0]`
    print 'maximum log marginal likelihood is: ' + `optimal_likelihood`
    # sample from test points, in other words, make prediction
    L_ = np.linalg.cholesky(K_ss + 1e-6 * np.eye(N) - np.dot(v.T, v))
    f_post_fun = mu_post.reshape(-1, 1) + np.dot(L_, np.random.normal(size=(N, num_fun)))
    plt.axis([-5, 5, -3, 3])
    return mu_post, stand_devi, f_post_fun, optimal_likelihood


def PI(params, means, stand_devi, parms_done, y, n_iterations, k):
    """
    Probability of Improvement acquisition function
    :param params: test data
    :param means: GP posterior mean
    :param stand_devi: standard deviation
    :param parms_done: training data
    :param y: training targets
    :return: next point that need to pick up
    """
    s = 0.0005  # small value
    stop_threshold = 0.001
    max_mean = np.max(y)

    f_max = max_mean + s
    z = (means - f_max) / stand_devi
    cumu_gaussian = norm.cdf(z)
    # early stop criterion, if there are less than 1% to get the greater cumulative Gaussian, then stop.
    if cumu_gaussian.sum() <= stop_threshold or np.max(cumu_gaussian) <= stop_threshold:
        print "all elements of cumulative are alost zeros!!!"
        return True
    indices = np.where(cumu_gaussian == np.max(cumu_gaussian))[0]
    indices = np.asarray(indices)
    # since there are several 1, random pick one of them as the next point except for parms_done
    rand_index = random.randint(0, len(indices) - 1)
    next_point = params[indices[rand_index]]
    condition = next_point in parms_done.tolist()
    # early stop criterion, if there is no other point that can maximize the objective then stop
    while condition:
        rand_index = random.randint(0, len(indices) - 1)
        next_point = params[indices[rand_index]]
        condition = next_point in parms_done.tolist()
        if len(next_point) == 1 and condition:
            return True

    if k == n_iterations-1:
        plt.subplot(2, 1, 2)
        plt.plot(params, cumu_gaussian)
        #plt.axis([1,5,-0.5,1.5])
    #plt.show()
    return next_point


def UCB(parms_done, params, means, stand_devi, n_iterations, k):
    """
    Upper Confidence Bound acquisition function
    :param params: test data
    :param means: GP posterior mean
    :param stand_devi: standard deviation
    :param parms_done: training data
    :return: next point that need to pick up
    """
    num_parms = len(parms_done)
    kappa = 0.001

    objective = means + kappa * stand_devi
    indices = np.where(objective == np.max(objective))[0]
    indices = np.asarray(indices)
    next_point = params[indices[0]]
    # if the next selected point is as same as the last one, then stop it
    if parms_done[num_parms - 1] == next_point:
        return True

    if k == n_iterations-1:
        plt.subplot(2, 1, 2)
        plt.plot(params, objective)
        #plt.axis([1,5,-0.5,1.5])
    return next_point


def TS(parms_done, params, y, n_iterations, k):
    """
    Thompson Sampling acquisition function
    :param parms_done: training data
    :param params: test data
    :param y: training targets
    :return: next point that need to pick up
    """
    num_fun = 1
    mu_post, stand_devi, f_post_fun = prediction(parms_done.reshape(-1, 1), params, y, 'rbf', 1, num_fun)
    max_post_fun = np.max(f_post_fun)
    max_index = np.where(f_post_fun == max_post_fun)
    next_point = params[max_index]
    if k == n_iterations-1:
        plt.subplot(2, 1, 2)
        plt.plot(params, f_post_fun)
    return next_point


def EI(params, means, stand_devi, parms_done, y, n_iterations, k):
    """
    Expected Improvement acquisition function
    :param params: test data
    :param means: GP posterior mean
    :param stand_devi: standard deviation
    :param parms_done: training data
    :param y: training targets
    :return: next point that need to pick up
    """
    s = 0.0005  # small value
    max_mean = np.max(y)

    f_max = max_mean + s
    z = (means - f_max) / stand_devi
    EI_vector = (means - f_max) * norm.cdf(z) + stand_devi * norm.pdf(z)
    max_index = np.where(EI_vector == np.max(EI_vector))
    next_point = params[max_index]
    if k == n_iterations-1:
        plt.subplot(2, 1, 2)
        plt.plot(params, EI_vector)
    return next_point

def acquisition_fun(params, means, stand_devi, parms_done, y, n_iterations, k):
    """
    different acquisition functions
    :param params: test data
    :param means: GP posterior mean
    :param stand_devi: standard deviation
    :param parms_done: training data
    :param y: training targets
    :return: next point that need to pick up
    """
    next_point_PI = PI(params, means, stand_devi, parms_done, y, n_iterations, k)
    next_point_UCB = UCB(parms_done, params, means, stand_devi, n_iterations, k)
    next_point_TS = TS(parms_done, params, y, n_iterations, k)
    next_point_EI = EI(params, means, stand_devi, parms_done, y, n_iterations, k)
    return next_point_PI#, next_point_UCB, next_point_TS, next_point_EI


def compute_mar_likelihood(X_train, X_test, y_train, sigma, l):
    """
    compute log marginal likelihood for tuning parameters using Bayesian optimization
    :param X_train: training data
    :param X_test: test data
    :param y_train: training targets
    :param sigma: output variance
    :param l: lengthscalar
    :return: log marginal likelihood
    """
    s = 0.0005  # noise variance and zero mean for noise
    n = len(X_train)

    # choose RBF kernel in this regression case
    K_train = RBF_kernel(X_train, X_train, sigma, l)
    K_s = RBF_kernel(X_train, X_test, sigma, l)
    K_ss = RBF_kernel(X_test, X_test, sigma, l)
    L = np.linalg.cholesky(K_train + s * np.eye(n))
    m = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, m)

    # compute mean of test points for posterior
    mu_post = np.dot(K_s.T, alpha)
    v = np.linalg.solve(L, K_s)

    # compute variance for test points
    var_test = np.diag(K_ss) - np.sum(v ** 2, axis=0)
    stand_devi = np.sqrt(var_test)

    # compute log marginal likelihood
    log_marg_likelihood = -.5 * np.dot(y_train.T, alpha) - np.log(np.diagonal(L)).sum(0) - n / 2.0 * np.log(2 * np.pi)
    #log_marg_likelihood = -.5 * np.dot(y_train.T, alpha) - np.diagonal(L).sum(0) - n / 2 * np.log(2 * np.pi)
    return log_marg_likelihood


def overlap(a, b):
    """
    return the indices in a that overlap with b, also returns
    the corresponding index in b only works if both a and b are unique!
    This is not very efficient but it works
    :param a: the first numpy array
    :param b: the second numpy array
    :return: indices for both numpy arrays
    """
    bool_a = np.in1d(a,b)
    ind_a = np.arange(len(a))
    ind_a = ind_a[bool_a]
    ind_b = np.array([np.argwhere(b == a[x]) for x in ind_a]).flatten()
    return ind_a,ind_b

def random_gen_test_parms(n, parms_done):
    """
    randomly generate test hyperparameters
    :param n: number of test hyperparameters you need
    :param parms_done: hyperparms that have already chosen
    :return: test hyperparameters
    """
    num_gen = n + len(parms_done) + 10
    test_parms = np.linspace(0.01, 5, num_gen)
    # remove duplicates between training and test points and then sample from non-duplicates test points
    ind_done, ind_sample = overlap(parms_done, test_parms)
    test_parms = np.delete(test_parms, ind_sample)
    test_parms_sampled = random.sample(test_parms, n)
    test_parms_sampled = np.asarray(test_parms_sampled)
    sample_sort = np.sort(test_parms_sampled)
    return sample_sort.reshape(-1, 1)


def tune_hyperparms_second(X_train, X_test, y_train, num_fun, sigma, l):
    """
    process of using Bayesian optimization to tune hyperparameters
    :param X_train: training data
    :param X_test: test data
    :param y_train: training targets
    :param num_fun: number of functions
    :param sigma: output variance
    :param l: lengthscalar
    :return: maximal log marginal likelihood
    """
    n = 100 # number of test points
    #next_point = np.zeros(4)
    n_iterations = 3

    for k in range(n_iterations):
        # randomly generate test hyperparms except for hyperparms that have already choosed
        l_test = random_gen_test_parms(n, l)
        log_marg_likelihood = np.zeros(len(l))
        for i in range(len(l)):
            log_marg_likelihood[i] = compute_mar_likelihood(X_train, X_test, y_train, sigma, l[i])
        # Bayesian optimization for hyperparameters
        mu_post, stand_devi, f_post_fun = bayesian_opt(l.reshape(-1,1), l_test, log_marg_likelihood)
        # determine the next training point using acquisition function
        next_point = acquisition_fun(l_test, mu_post, stand_devi, l, log_marg_likelihood, n_iterations, k)
        #print next_point
        # if get early stop criterion then stop and output optimal result
        if next_point is True:
            max_index = np.where(log_marg_likelihood == np.max(log_marg_likelihood))[0]
            print "it takes " + `k+1` + " iterations to get the optimal!"
            print "optimal lenghscalar is:" + `l[max_index][0]`
            break
        l = np.append(l, next_point)
        max_index = np.where(log_marg_likelihood == np.max(log_marg_likelihood))[0]

    log_marg_likelihood = np.zeros(len(l))
    for i in range(len(l)):
        log_marg_likelihood[i] = compute_mar_likelihood(X_train, X_test, y_train, sigma, l[i])
    print "it takes " + `k+1` + " iterations to get the optimal!"
    print "optimal lenghscalar is:" + `l[max_index][0]`
    print "maximum likelihood is:" + `np.max(log_marg_likelihood)`
    # Bayesian optimization for hyperparameters
    l_test = random_gen_test_parms(n, l)
    mu_post, stand_devi, f_post_fun = bayesian_opt(l.reshape(-1, 1), l_test, log_marg_likelihood)
    # plot Bayesian optimization
    plot_BO(l.reshape(-1,1), log_marg_likelihood, l_test, f_post_fun, mu_post, stand_devi)
    return np.max(log_marg_likelihood)


def tune_hyperparms_gradient(X_train, X_test, y_train, num_fun):
    """
    tune hyperparameters by maximizing the log marginal likelihood
    :param X_train: training data
    :param X_test: test data
    :param y_train: training targets
    :param num_fun: number of functions
    :return: maximal log marginal likelihood
    """
    sigma = 1 # fix the output variance of RBF kernel
    l = np.random.uniform(0, 5, 1)  # randomly initial hyperparmeters
    # tune hyperparameters of RBF kernel in the regression using gradient ascent
    mu_post, stand_devi, f_post_fun, optimal_likelihood = tune_hyperparms_first(X_train, X_test, y_train, num_fun, sigma, l)
    # plot posterior functions
    plot_posterior(X_test, f_post_fun, mu_post, stand_devi)
    # plot true function and difference between true function and posterior prediction
    plot_true_diff(X_train, X_test, y_train, true_fun, mu_post, stand_devi)
    return optimal_likelihood


def tune_hyperparms_BO(X_train, X_test, y_train, num_fun):
    """
    initialization of tuning hyperparameters using Bayesian optimization
    :param X_train:
    :param X_test:
    :param y_train:
    :param num_fun:
    :return:
    """
    sigma = 1 # fix the output variance of RBF kernel
    # random pick up two initial hyperparm
    l = np.random.uniform(0.02,5,2) # np.array([0.5, 3.5])
    # tune hyperparameters of RBF kernel in regression using Bayesian optimization
    optimal_likelihood = tune_hyperparms_second(X_train, X_test, y_train, num_fun, sigma, l)
    return optimal_likelihood


if __name__ == "__main__":
    N = 3  # number of training points
    n = 100  # number of test points
    sigma = 1 # fix the output variance of RBF kernel
    num_fun = 10  # number of prior function

    # generate dataset for GP regression
    true_fun, X_train, y_train, X_test = dataset_generator(N, n)

    # using Bayesian optimization in order to tune the hyperparameters
    print ""
    print "------ Bayesian oprimization ------"
    optimal_likelihood_BO = tune_hyperparms_BO(X_train, X_test, y_train, num_fun)

    # using gradient ascent to maximize log marginal likelihood in order to tune the hyperparameters
    print ""
    print "------ gradient ascent------"
    optimal_likelihood_GA = tune_hyperparms_gradient(X_train, X_test, y_train, num_fun)


    # the difference between result of Bayesian optimization and gradient ascent
    error = np.abs(optimal_likelihood_BO - optimal_likelihood_GA) / \
            max(np.abs(optimal_likelihood_BO), np.abs(optimal_likelihood_GA))
    error_rate = error*100
    print ""
    print "------ error rate ------"
    print("The error rate of optimal likelihood between two methods is: %.3f%%" % error_rate)