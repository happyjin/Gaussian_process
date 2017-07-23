import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)


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


if __name__ == "__main__":
    info = np.load('data_list.npy')
    # get positive data
    pos_data = get_pos_data(info)

    # plot CO2
    plot_CO2(pos_data)

    plt.show()