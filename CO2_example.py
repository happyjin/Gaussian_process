from tune_hyperparms_regression import overlap
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

def kernel_1(sqdist, theta_1, theta_2):
    """
    the first kernel--the RBF kernel
    :param sqdist: input vector a
    :param theta_1: input vector b
    :param theta_2: lengthscale
    :return: kernel matrix(covariance matrix)
    """
    return (theta_1 ** 2) * np.exp(-.5 * (1 / (theta_2 ** 2)) * sqdist)


def kernel_2(a, b, theta_3, theta_4, theta_5):
    """
    the second kernel--the modified periodic kernel
    :param a: the first input vector
    :param b: the second input vector
    :param theta_3: the magnitude
    :param theta_4: the decay-time for the periodic component
    :param theta_5: the smoothness of the periodic component
    :return: covariance matrix
    """
    num_a = len(a)
    num_b = len(b)
    sqdist = ((a[:, :, None] - b[:, :, None].T) ** 2).sum(1)
    l2_norm = np.absolute(np.tile(a, (1, num_b)) - np.tile(b.T, (num_a, 1)))
    first_item = -.5 * sqdist / theta_4**2
    second_item = -2 * np.sin(np.pi * l2_norm)**2 / theta_5**2
    return theta_3 * np.exp(first_item + second_item)


def kernel_3(sqdist, theta_6, theta_7, theta_8):
    """
    the third kernel--the rational quadratic kernel in order to model the (small) medium term irregularities
    :param sqdist: l2-norm square
    :param theta_6: the magnitude
    :param theta_7: the typical length-scale
    :param theta_8: the shape parameter
    :return: covariance matrix
    """
    item = 1 + .5 * sqdist / (theta_8 * theta_7**2)
    part_kernel = 1.0 / np.power(item, theta_8)
    return theta_6**2 * part_kernel


def kernel_4(sqdist, theta_9, theta_10, theta_11):
    """
    the forth kernel--a noise model
    :param sqdist: l2-norm square
    :param theta_9: the magnitude of the correlated noise component
    :param theta_10: lengthscale
    :param theta_11: the magnitude of the independent noise component
    :return: covariance noise matrix
    """
    delta = 0.5
    item = np.exp(-.5 * sqdist / theta_10**2)
    return theta_9**2 * item + theta_11**2 * delta


def covariance_function(a, b, hyperparms):
    sqdist = ((a[:, :, None] - b[:, :, None].T) ** 2).sum(1)
    kernel = kernel_1(sqdist, hyperparms[0], hyperparms[1]) + \
             kernel_2(a, b, hyperparms[2], hyperparms[3], hyperparms[4]) + \
             kernel_3(sqdist, hyperparms[5], hyperparms[6], hyperparms[7]) + \
             kernel_4(sqdist, hyperparms[8], hyperparms[9], hyperparms[10])
    return kernel

def get_pos_data(info):
    """
    remove outlier and get positive data
    :param info: all information
    :return: positive data
    """
    data = info[:, 1:13].flatten()
    neg_indices = np.where(data <= 0)
    pos_data = np.delete(data, neg_indices)
    return pos_data


def plot_CO2(X_train, y_train):
    """
    plot CO2 diagram
    :param X_train: training data
    :param y_train: training targets
    :return:
    """
    num_data = len(y_train)

    # plot data
    plt.plot(X_train, y_train)


def random_sample_test_parms(n_test_hyperparms, train_parms):
    """
    randomly sample hyperparameters for test
    :param n_test_hyperparms: number of test hyperparameters
    :param train_parms: training hyperparms, in other words, params_done
    :return: test hyperparameters matrix
    """
    test_parms_matrix = np.zeros(shape=(len(train_parms), n_test_hyperparms))
    for i in range(len(train_parms)):
        num_gen = n_test_hyperparms + len(train_parms[i]) + 10
        test_parms = np.linspace(0.01, 150, num_gen)
        # remove duplicates between training and test points and then sample from non-duplicates test points
        ind_done, ind_sample = overlap(train_parms[i], test_parms)
        #print ind_done
        #print ind_sample
        test_parms = np.delete(test_parms, ind_sample)
        test_parms_sampled = random.sample(test_parms, n_test_hyperparms)
        test_parms_sampled = np.asarray(test_parms_sampled)
        sample_sort = np.sort(test_parms_sampled)
        test_parms_matrix[i] = np.copy(sample_sort)
    return test_parms_matrix


def compute_mar_likelihood(X_train, X_test, y_train, hyperparms):
    s = 0.0005  # noise variance and zero mean for noise
    N = len(X_train)
    n = len(X_test)

    K_train = covariance_function(X_train, X_train, hyperparms)
    K_s = covariance_function(X_train, X_test, hyperparms)
    K_ss = covariance_function(X_test, X_test, hyperparms)
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
    return log_marg_likelihood

def tune_hyperparameters_BO(X_train, y_train, X_test):
    train_hyperparms = np.zeros(shape=(3, 11)) # define initial hyperparms
    for i in range(3):
        train_hyperparms[i] = np.random.uniform(0.01, 150, 11) # randomly initial hyperparameters
    #print hyperparms.shape
    n_test_hyperparms = 100  # number of test hyperparameters

    n_train_data = len(X_train)
    n_train_hyperparms = len(train_hyperparms)
    test_hyperparms = random_sample_test_parms(n_test_hyperparms, train_hyperparms)
    log_marg_likelihood = np.zeros(n_train_data)
    for i in range(n_train_hyperparms):
        log_marg_likelihood[i] = compute_mar_likelihood(X_train, X_test, y_train, train_hyperparms[i])
    # Bayesian optimization for hyperparameters



if __name__ == "__main__":
    info = np.load('data_list.npy')
    # preprocess data
    pos_data = get_pos_data(info)
    empirical_mean = np.mean(pos_data)
    y_train = pos_data - empirical_mean
    X_train = (np.arange(len(y_train))).reshape(-1,1)
    X_test = (np.arange(np.max(X_train)+1, np.max(X_train)+101)).reshape(-1,1)

    # plot CO2
    plot_CO2(X_train, y_train)
    #plt.show()

    # compute covariance function
    #kernel = covariance_function(X_train, X_train)

    # tune hyperparameters
    tune_hyperparameters_BO(X_train.reshape(-1,1), y_train, X_test)