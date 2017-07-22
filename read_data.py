import pickle
import re
import os.path
import numpy as np
#import deepdish as dd

def save_data(file):
    """
    save data
    :param file: file name
    :return:
    """
    # read data
    data = []
    fp = open(file)
    for i, line in enumerate(fp):
        if i >= 15 and i <= 65:
            print line
            data.append(line)
    fp.close()
    # save data
    with open('CO2_data', 'wb') as fp:
        pickle.dump(data, fp)


def generate_data_list(itemlist):
    """
    convert str data list into float data list
    :param itemlist: str data list
    :return: float data list
    """
    data = []
    for i in range(len(itemlist)):
        item = re.split("[\\t\\n]", itemlist[i])
        data.append(item)

    data_list = []
    for k in range(len(data)):
        data_sub_list = []
        for i in range(len(data[0])-1):
            if data[k][i].find('-') != -1:
                abs_digit = data[k][i].split("-", 1)[1]
                abs_digit = float(abs_digit)
                data_sub_list.append(-abs_digit)
            else:
                data_sub_list.append(float(data[k][i]))
        data_list.append(data_sub_list)
    return data_list


def save_data_narray(data_list):
    """
    save float data list
    :param data_list: float data list
    :return:
    """
    data_list = np.asarray(data_list)
    for i in range(len(data_list)):
        data_list[i] = np.asarray(data_list[i])
    append_narray = np.append([data_list[0]], [data_list[1]], axis=0)
    for i in range(len(data_list) - 2):
        append_narray = np.append(append_narray, [data_list[i + 2]], axis=0)
    save_name = 'data_list.npy'
    if not os.path.isfile(save_name):
        np.save(save_name, data_list)
    return data_list

if __name__ == "__main__":
    file = "data.txt"

    # if file not exists then create it
    if not os.path.isfile(file):
        save_data(file)

    # load data
    data_name = 'CO2_data'
    with open(data_name, 'rb') as fp:
        itemlist = pickle.load(fp)

    # convert str data list into float data list
    data_list = generate_data_list(itemlist)

    # save float data list
    data_narray = save_data_narray(data_list)

    # load data narray
    data_narray = np.load('data_list.npy')