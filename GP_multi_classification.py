#from __future__ import division
from sklearn.datasets import make_blobs
from sklearn.cross_validation import train_test_split
from GP_regression import RBF_kernel
from scipy.linalg import block_diag
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=3, suppress=True, threshold=np.nan)


def savetxt_compact(fname, x, fmt="%1.2f", delimiter=' '):
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


def compute_pi(f, pi_vector, pi_matrix, C, n):
    """
    compute pi_matrix and pi_vector
    :param f: functions
    :param pi_vector: column vector of pi elements
    :param pi_matrix: Cn*n matrix by stacking vertically the diagonal matrices pi_c
    :param C: num of categories
    :param n: num of training data
    :return: pi_vector, pi_matrix
    """
    softmax_output = np.zeros(C)
    index_put = np.arange(3)

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
        index_put += 3
    #np.savetxt('pi_matrix.txt',pi_matrix, fmt='%1.2f', delimiter=' ', newline=os.linesep)
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
    # initialization
    f = np.zeros((C*n,)) # initialize f=0(unbiased) which is an constant=0 function and means no GP prior in this case
    pi_vector = np.zeros_like(f)
    pi_matrix = np.zeros((C*n, n))
    D = np.zeros((C*n, C*n))
    tolerance = 0.0001
    s = 0.0005

    # Newton iteration
    for i in range(1):
        pi_vector, pi_matrix = compute_pi(f, pi_vector, pi_matrix, C, n)
        np.fill_diagonal(D, pi_vector)
        #np.savetxt('D.txt',D, fmt='%1.2f', delimiter=' ', newline=os.linesep)
        M = n
        block_K = [K[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K.shape[0] / M)]
        block_D = [D[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(D.shape[0] / M)]
        E_c = np.zeros((n, n))
        for c in range(C):
            L = np.linalg.cholesky(np.eye(n) + np.dot(np.sqrt(block_D[c]), np.dot(block_K[c], block_D[c])))
            L_inv = np.linalg.inv(L)
            E_c = np.dot(np.sqrt(block_D[c]), np.dot(L_inv.T, np.dot(L_inv, np.sqrt(block_D[c]))))
            # create a block diagonal matrix E
            if c == 0:
                E = E_c
            else:
                E = block_diag(E, E_c)
            E_c += E_c
        M = np.linalg.cholesky(E_c)
        b = np.dot(D - np.dot(pi_matrix, pi_matrix.T), f) + y - pi_vector
        c = np.dot(E, np.dot(K, b))
        R = np.dot(np.linalg.inv(D), pi_matrix)
        M_inv = np.linalg.inv(M)
        a = b - c + np.dot(E, np.dot(R, np.dot(M_inv.T, np.dot(M_inv, np.dot(R.T, c)))))
        f_new = np.dot(K, a)

        error = np.sqrt(np.sum((f_new - f) ** 2))
        f = f_new
        print `i + 1` + "th iteration, error:" + `error`
        if error <= tolerance:
            print "The function has already converged after " + `i + 1` + " iterations!"
            print "The error is " + `error`
            print "training end!"
            break




def dataset_generator():
    """
    generate multi-class dataset
    :return:
    """
    plt.title("Three blobs", fontsize='small')
    X, Y = make_blobs(n_features=2, centers=3)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
    plt.show()
    np.save('X_multi.npy', X)
    np.save('y_multi.npy', Y)


if __name__ == "__main__":
    if not os.path.exists('X_multi.npy'):
        dataset_generator()
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
    K1 = RBF_kernel(X_train, X_train, kernel_parameter)
    K = np.kron(np.eye(num_classes), K1)
    savetxt_compact('K.txt', K)
    # generate 0/1 targets for training dataset
    y_targets = np.zeros((num_classes*num_train,))
    index = np.arange(num_train)
    #for i in range(len(num_classes)):
    indices = y_train * 60 + index
    y_targets[indices] = 1

    # train the model
    model_training(K, y_targets, num_classes, num_train)
