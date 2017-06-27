from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import expit # logistic function
from GP_regression import RBF_kernel, f_prior
from scipy.stats import norm
np.set_printoptions(precision=3, suppress=True, threshold=np.nan)


# generate dataset

def dataset_generator():
    """

    :return:
    """
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]

    X, y = datasets[0]
    y[y == 0] = -1
    X = StandardScaler().fit_transform(X)
    return X, y

def label_function(f_star):
    """
    get either test point belongs to +1 or -1 label
    :param f_star: function value of x_star
    :return: label for this test point
    """
    prob_label = pi_function(f_star)
    if prob_label >= 0.5:
        return 1
    else:
        return -1


def pi_function(f):
    """
    deterministic function
    :param f: function value e.g. f(x1), f(x2) etc
    :return: logistic function
    """
    return expit(f)


def log_likelihood(z):
    """
    # log_likelihood function
    :param z: label times its function value
    :return: its log likelihood of current funciton
    """
    return -np.log(1 + np.exp(-z))


def deriv_log_likelihood(y, f):
    """
    first derivative of log_likelihood function
    :param y: label
    :param f: function value
    :return: first derivative of its log likelihood function
    """
    t = (y + 1) / 2
    return t - pi_function(y*f)


def sec_deriv_log_likelihood(f):
    """
    second derivative of log_likelihood function
    :param f: function
    :return: its second derivative of log likelihood
    """
    return -pi_function(f) * (1 - pi_function(f))


def newton_method(K, y_train, f_prior, num_funs):
    """
    Newton method update function in order to find the mode of Gaussian approx.
    :param K: kernel matrix
    :param y_train: label for training dataset
    :param f_prior: GP prior
    :param num_funs: number of prior functions to use
    :return: W, L inversion, first derivative at optimal point, in other words, mode
    """
    num_train = y_train.size
    W = np.zeros((num_train, num_train))
    f = np.zeros((num_train, num_funs))  # initialize function####

    tolerance = 0.0001

    print "training model!"
    for i in range(10000):
        first_deri = deriv_log_likelihood(y_train, f_prior)
        np.fill_diagonal(W, -sec_deriv_log_likelihood(f_prior))

        L = np.linalg.cholesky(np.eye(num_train) + np.dot(np.dot(np.sqrt(W), K), np.sqrt(W)))
        L_inv = np.linalg.inv(L)
        b = np.dot(W, f) + first_deri########
        a = b - np.dot(np.sqrt(W), np.dot(L_inv.T, np.dot(L_inv, np.dot(np.dot(np.sqrt(W), K), b))))
        f_new = np.dot(K, a)

        error = np.sqrt(np.sum((f_new - f)**2))
        f = f_new
        print `i+1` + "th iteration, error:" + `error`
        if error <= tolerance:
            print "The function has already converged after " + `i+1` + " iterations!"
            print "The error is " + `error`
            print "training end!"
            break

    return W, L_inv, first_deri


def prediction(x_star, y_star_true, X_train, L_inv, W, first_deri, kernel_parameter):
    """
    make prediction
    :param x_star: input test point
    :param y_star_true: its label
    :param X_train: training dataset
    :param L_inv: L inversion after Cholesky decomposition
    :param W:
    :param first_deri: first derivative at mode(optimal point after Newton update)
    :param kernel_parameter: specify kernel parameter
    :return: prediction label
    """
    k_star = RBF_kernel(X_train, x_star, kernel_parameter)
    f_star_mean = np.dot(k_star.T, first_deri)
    v = np.dot(L_inv, np.dot(np.sqrt(W), k_star))
    k_ss = RBF_kernel(x_star, x_star, kernel_parameter)
    var_f_star = k_ss - np.dot(v.T, v)
    return label_function(f_star_mean) == y_star_true




if __name__ == "__main__":

    # generate dataset
    X, y = dataset_generator()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    num_train = len(X_train)
    num_test = len(X_test)

    # hyper-parameters
    num_funs = 1 # number of GP prior functions
    # number of sampling arbitrary points for GP prior. For simplicity we assume num_sampling is equal to num_train
    num_sampling = num_train
    kernel_parameter = 1

    # plot dataset scatter at first
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.subplot(2,3,1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    plt.title('data points')

    # compute covariance matrix K under RBF_kernel.
    K_train = RBF_kernel(X_train, X_train, kernel_parameter)

    # sampling points for GP prior function
    x1_min = np.min(X[:,0])
    x1_max = np.max(X[:,0])
    x2_min = np.min(X[:,1])
    x2_max = np.max(X[:,1])
    X1_sampling = np.linspace(x1_min, x1_max, num_sampling).reshape(-1, 1)
    X2_sampling = np.linspace(x2_min, x2_max, num_sampling).reshape(-1, 1)
    X_sampling = np.concatenate((X1_sampling, X2_sampling), axis=1)

    # sampling GP prior sampling GP prior f for likelihood
    mu_prior = np.zeros((num_sampling, 1))
    num_sampling = X_sampling.shape[0]
    f_prior = f_prior(X_sampling, mu_prior, kernel_parameter, num_sampling, num_funs)
    print 'shape of prior function:' + `f_prior.shape`
    plt.subplot(2,3,2)
    #plt.clf()
    plt.plot(np.sort(X_sampling[:,0]), f_prior)
    plt.title('latent function-prior')

    # compute probability of GP prior under y=+1 compute likelihood function p(y|f)
    y = +1
    prob_pred_prior = expit(y * f_prior)
    plt.subplot(2,3,3)
    #plt.clf()
    plt.plot(np.sort(X_sampling[:,0]), prob_pred_prior)
    plt.title('probability for latent function-prior')
    #plt.show()
    # deterministic function
    #print y_pred >0.5

    # compute likelihood function p(y|f), we fix y(training label) and vary GP prior function f
    y_train = y_train.reshape(-1, 1)
    y_label = 1
    z = +1 * f_prior
    log_likeli = log_likelihood(z)
    deriv_log_likeli = deriv_log_likelihood(y_label, f_prior)
    sec_deriv_log_likeli = sec_deriv_log_likelihood(f_prior)

    # log likelihood and its derivatives figure
    plt.subplot(2,3,4)
    #plt.clf()
    i = 0
    plt.plot(z[:,i], log_likeli[:,i], 'b-', label='log likelihood')
    plt.plot(z[:,i], deriv_log_likeli[:,i], 'r--', label='1st derivative')
    plt.plot(z[:,i], sec_deriv_log_likeli[:,i], 'g--', label='2nd derivative')
    plt.legend(loc=4)

    #plt.show()

    i = 4
    # newton iteration
    W, L_inv, first_deri = newton_method(K_train, y_train, f_prior, num_funs, X_test[i].reshape(-1,2), y_test[i])
    true_count = np.ones(num_test)
    for i in range(len(X_test)):
        judgement = prediction(X_test[i].reshape(-1,2), y_test[i], X_train, L_inv, W, first_deri, kernel_parameter)
        if not judgement:
            true_count[i] = 0

    false_binary_index = np.where(true_count == 0)[0]

    print "classification right rate is: %0.2f" %(true_count.sum(0) / len(X_test) * 100)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.subplot(2, 3, 5)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    for i in range(len(false_binary_index)):
        plt.scatter(X_test[false_binary_index[i], 0], X_test[false_binary_index[i], 1], marker='x', color='c')
    plt.title('results with false data points')
    plt.show()