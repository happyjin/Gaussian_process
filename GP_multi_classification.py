from sklearn.datasets import make_blobs
from sklearn.cross_validation import train_test_split
from GP_regression import RBF_kernel
from scipy.linalg import block_diag
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=3, suppress=True, threshold=np.nan)


def savetxt_compact(fname, x, fmt="%1.1f", delimiter=' '):
    """
    save matrix or vector into txt in order to visualize the data
    :param fname: name of file, e.g. matrix.txt
    :param x: data
    :param fmt: format of data
    :param delimiter: format between each element
    :return:
    """
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')


def softmax(X):
    """
    Compute softmax values for each sets of scores in x
    :param X:
    :return:
    """
    e_x = np.exp(X - np.max(X))
    return e_x / e_x.sum(axis=0)


def compute_pi(f, C, n):
    """
    compute pi_matrix and pi_vector through softmax function
    :param f: functions
    :param pi_vector: column vector of pi elements
    :param pi_matrix: Cn*n matrix by stacking vertically the diagonal matrices pi_c
    :param C: num of categories
    :param n: num of training data
    :return: pi_vector, pi_matrix
    """
    pi_vector = np.zeros_like(f)
    pi_matrix = np.zeros((C*n, n))
    softmax_output = np.zeros(C)
    index_put = np.arange(C)

    for i in range(n):
        triple = np.zeros(C)
        pi_column = np.zeros(C*n)
        for j in range(C):
            triple[j] = f[j*60+i]
        for j in range(C):
            softmax_output = softmax(triple)
            pi_vector[j*60+i] = softmax_output[j]
        pi_column.put(index_put, softmax_output)
        pi_matrix[:, i] = np.copy(pi_column)
        index_put += C

    return pi_vector, pi_matrix


def model_training(K, y, C, n):
    """
    train the model
    :param K: kernel matrix
    :param y: labels which has the same length as all functions f
    :param C: num of categories
    :param n: num of training data
    :return:
    """
    tolerance = 0.0001
    step_size = 0.00001
    s = 0.0005
    # initialization
    f = np.zeros((C*n,)) # initialize f=0(unbiased) which is an constant=0 function and means no GP prior in this case
    block_K = [K[i * n:(i + 1) * n, i * n:(i + 1) * n] for i in range(K.shape[0] / n)]

    # Newton iteration
    for j in range(100):
        pi_vector, pi_matrix = compute_pi(f, C, n)
        D = np.zeros((C * n, C * n))
        np.fill_diagonal(D, pi_vector)
        savetxt_compact('D.txt',D)
        block_D = [D[i * n:(i + 1) * n, i * n:(i + 1) * n] for i in range(D.shape[0] / n)]
        savetxt_compact('block_D0.txt', block_D[0])
        savetxt_compact('block_D2.txt', block_D[2])
        E_c_sum = np.zeros((n, n))
        for c in range(C):
            L = np.linalg.cholesky(np.eye(n) + np.dot(np.sqrt(block_D[c]), np.dot(block_K[c], np.sqrt(block_D[c]))))
            L_inv = np.linalg.inv(L)
            E_c_part = np.dot(np.sqrt(block_D[c]), np.dot(L_inv.T, np.dot(L_inv, np.sqrt(block_D[c]))))
            # create a block diagonal matrix E
            if c == 0:
                E = E_c_part
            else:
                E = block_diag(E, E_c_part)
            E_c_sum += E_c_part
        L_whole = np.linalg.cholesky(np.eye(C*n) + np.dot(np.sqrt(D), np.dot(K, np.sqrt(D))))
        L_whole_inv = np.linalg.inv(L_whole)
        E = np.dot(np.sqrt(D), np.dot(L_whole_inv.T, np.dot(L_whole_inv, np.sqrt(D))))
        #E = np.dot(np.sqrt(D), np.dot(np.linalg.inv(np.eye(C*n) + np.dot(np.sqrt(D), np.dot(K, np.sqrt(D)))), np.sqrt(D)))
        R = np.dot(np.linalg.inv(D), pi_matrix)
        #M = np.linalg.cholesky(E_c_sum)
        M = np.linalg.cholesky(np.dot(R.T, np.dot(E, R)))
        W = D - np.dot(pi_matrix, pi_matrix.T)
        L_K = np.linalg.cholesky(s * np.eye(C*n) + K)
        L_K_inv = np.linalg.inv(L_K)

        b = np.dot((1-step_size) * np.dot(L_K_inv.T, L_K_inv) + W, f) + y - pi_vector
        c = np.dot(E, np.dot(K, b))
        M_inv = np.linalg.inv(M)
        a = b - c + np.dot(E, np.dot(R, np.dot(M_inv.T, np.dot(M_inv, np.dot(R.T, c)))))
        f_new = np.dot(K, a)

        error = np.sqrt(np.sum((f_new - f) ** 2))
        f = f_new
        print `j + 1` + "th iteration, error:" + `error`
        if error <= tolerance:
            print "The function has already converged after " + `j + 1` + " iterations!"
            print "The error is " + `error`
            print "training end!"
            break


def model_training2(K, y, C, n):
    """
    train the model
    :param K: kernel matrix
    :param y: labels of training dataset
    :param C: num of classes
    :param n: num of training data
    :return: pi vector which computed through softmax
    """
    tolerance = 0.01
    step_size = 0.0001
    s = 3
    # initialization
    f = np.zeros((C * n,))  # initialize f=0(unbiased) which is an constant=0 function and means no GP prior in this case

    # Newton iteration
    for j in range(10000):
        pi_vector, pi_matrix = compute_pi(f, C, n)
        L = np.linalg.cholesky(s * np.eye(C*n) + K)
        L_inv = np.linalg.inv(L)
        D = np.zeros((C * n, C * n))
        np.fill_diagonal(D, pi_vector)
        W = D - np.dot(pi_matrix, pi_matrix.T)
        sec_deri = np.dot(L_inv.T, L_inv) + W
        print sec_deri.shape
        L_sec_deri = np.linalg.cholesky(s * np.eye(C*n) + sec_deri)
        L_inv_sec_deri = np.linalg.inv(L_sec_deri)
        sum = np.dot((1-step_size)*np.dot(L_inv.T, L_inv)+W, f) + y + pi_vector
        f_new = np.dot(L_inv_sec_deri, sum)
        error = np.sqrt(np.sum((f_new - f) ** 2))
        f = f_new
        print `j + 1` + "th iteration, error:" + `error`
        if error <= tolerance:
            print "The function has already converged after " + `j + 1` + " iterations!"
            print "The error is " + `error`
            print "training end!"
            break
    return pi_vector


def prediction(x_star, y_star_true, X_train, C, y, pi_vector, kernel_parameter):
    """
    make prediction
    :param x_star: test input
    :param y_star_true: label of test input
    :param X_train: training dataset
    :param C: num of classes
    :param y: labels of training dataset
    :param pi_vector: pi vector which computed through softmax function
    :param kernel_parameter: parameter for kernel
    :return: true or false
    """
    n = len(X_train)
    k_star = RBF_kernel(X_train, x_star, kernel_parameter)
    f_star_mean = np.zeros((C,))
    for c in range(C):
        f_star_mean[c] = np.dot(k_star.T, y[c*n:(c+1)*n] - pi_vector[c*n:(c+1)*n])
    return np.argmax(f_star_mean) == y_star_true


def dataset_generator():
    """
    generate multi-class dataset
    :return: data X and its labels
    """
    plt.title("Three blobs", fontsize='small')
    X, y = make_blobs(n_features=2, centers=3)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()
    #np.save('X_multi.npy', X)
    #np.save('y_multi.npy', y)
    return X, y


if __name__ == "__main__":
    if not os.path.exists('X_multi.npy'):
        # randomly generate dataset for classification task
        X, y = dataset_generator()
    else:
        X = np.load('X_multi.npy')
        y = np.load('y_multi.npy')
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    num_train = len(X_train)
    num_test = len(X_test)
    num_classes = np.size(np.unique(y))

    # hyper-parameters
    num_funs = num_classes # num of GP prior functions = num of categories
    kernel_parameter = 1

    # compute kernel matrix K
    for c in range(num_classes):
        K_sub = RBF_kernel(X_train, X_train, kernel_parameter)
        if c == 0:
            K = K_sub
        else:
            K = block_diag(K, K_sub)
    # generate 0/1 targets for training dataset
    y_targets = np.zeros((num_classes*num_train,))
    index = np.arange(num_train)
    indices = y_train * 60 + index
    y_targets[indices] = 1

    # train the model
    #model_training(K, y_targets, num_classes, num_train)
    pi_vector = model_training2(K, y_targets, num_classes, num_train)
    true_count = np.ones(num_test)
    for i in range(len(X_test)):
        judgement = prediction(X_test[i].reshape(-1,2), y_test[i], X_train, num_classes, y_targets, pi_vector, kernel_parameter)
        if not judgement:
            true_count[i] = 0
    print "classification right rate is: %0.2f percent" % (true_count.sum(0) / len(X_test) * 100)