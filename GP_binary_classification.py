import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import expit # logistic function
from GP_regression import RBF_kernel, f_prior
np.set_printoptions(precision=3, suppress=True, threshold=np.nan)

# generate dataset
def dataset_generator():
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


# log_likelihood function
def log_likelihood(z):
    return -np.log(1 + np.exp(z))


# first derivative of log_likelihood function
def deriv_log_likelihood(y, z):
    t = (y + 1) / 2
    return t - expit(z)


# second derivative of log_likelihood function
def sec_deriv_log_likelihood(z):
    return -expit(z) * (1 - expit(z))


# Newton method function
def newton_method(K, y, z, num_funs):
    num_ele = y.size
    first_deri = np.zeros((num_ele, num_ele))
    W = np.zeros((num_ele, num_ele))
    f = np.zeros((num_ele, num_funs)) # initialize function
    tolerance = 0.0000001


    for i in range(100):
        first_deri = deriv_log_likelihood(y, z)
        np.fill_diagonal(W, -sec_deriv_log_likelihood(z))
        L = np.linalg.cholesky(np.eye(num_ele) + np.dot(np.dot(np.sqrt(W), K), np.sqrt(W)))
        L_inv = np.linalg.inv(L)
        b = np.dot(W, f) + first_deri
        a = b - np.dot(np.sqrt(W), np.dot(L_inv.T, np.dot(L_inv, np.dot(np.dot(np.sqrt(W), K), b))))
        f_new = np.dot(K, a)
        difference = np.absolute(f_new - f)
        error = np.sqrt(np.sum(difference**2))
        z = y_train * f_new
        f = f_new
        print error

        if error <= tolerance:
            print "The function has already converged after " + `i+1` + " iterations!"
            print "The error is " + `error`
            break




if __name__ == "__main__":

    # generate dataset
    X, y = dataset_generator()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    num_train = y_train.size

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

    # compute covariance matrix K under RBF_kernel
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
    z = y_train * f_prior
    prob_likelihood = expit(z)
    log_likeli = log_likelihood(z)
    deriv_log_likeli = deriv_log_likelihood(y_train, z)
    sec_deriv_log_likeli = sec_deriv_log_likelihood(z)

    # log likelihood and its derivatives figure
    plt.subplot(2,3,4)
    #plt.clf()
    i = 0
    plt.plot(z[:,i], log_likeli[:,i], 'bo', label='log likelihood')
    plt.plot(z[:,i], deriv_log_likeli[:,i], 'ro', label='1st derivative')
    plt.plot(z[:,i], sec_deriv_log_likeli[:,i], 'go', label='2nd derivative')
    plt.legend()
    #plt.show()

    # newton iteration
    newton_method(K_train, y_train, z, num_funs)
