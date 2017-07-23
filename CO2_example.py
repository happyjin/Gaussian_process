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


def plot_CO2(pos_data):
    """
    plot CO2 diagram
    :param pos_data:
    :return:
    """
    num_data = len(pos_data)
    x_axis = np.arange(num_data)
    # plot data
    plt.plot(x_axis, pos_data)
    plt.show()


if __name__ == "__main__":
    info = np.load('data_list.npy')
    # get positive data
    pos_data = get_pos_data(info)

    # plot CO2
    plot_CO2(pos_data)