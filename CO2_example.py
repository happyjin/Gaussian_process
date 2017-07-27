from tune_hyperparms_regression import overlapimport sysimport randomimport numpy as npimport matplotlib.pyplot as pltnp.set_printoptions(precision=3, suppress=True)def kernel_1(sqdist, theta_1, theta_2):    """    the first kernel--the RBF kernel    :param sqdist: input vector a    :param theta_1: input vector b    :param theta_2: lengthscale    :return: kernel matrix(covariance matrix)    """    return (theta_1 ** 2) * np.exp(-.5 * (1 / (theta_2 ** 2)) * sqdist)def kernel_2(l2_norm, sqdist, theta_3, theta_4, theta_5):    """    the second kernel--the modified periodic kernel    :param a: the first input vector    :param b: the second input vector    :param theta_3: the magnitude    :param theta_4: the decay-time for the periodic component    :param theta_5: the smoothness of the periodic component    :return: covariance matrix    """    #num_a = len(a)    #num_b = len(b)    #sqdist = ((a[:, :, None] - b[:, :, None].T) ** 2).sum(1)    #l2_norm = np.absolute(np.tile(a, (1, num_b)) - np.tile(b.T, (num_a, 1)))    first_item = -.5 * sqdist / theta_4**2    second_item = -2 * np.sin(np.pi * l2_norm)**2 / theta_5**2    return theta_3 * np.exp(first_item + second_item)def kernel_3(sqdist, theta_6, theta_7, theta_8):    """    the third kernel--the rational quadratic kernel in order to model the (small) medium term irregularities    :param sqdist: l2-norm square    :param theta_6: the magnitude    :param theta_7: the typical length-scale    :param theta_8: the shape parameter    :return: covariance matrix    """    item = 1 + .5 * sqdist / (theta_8 * theta_7**2)    part_kernel = 1.0 / np.power(item, theta_8)    return theta_6**2 * part_kerneldef kernel_4(sqdist, theta_9, theta_10, theta_11):    """    the forth kernel--a noise model    :param sqdist: l2-norm square    :param theta_9: the magnitude of the correlated noise component    :param theta_10: lengthscale    :param theta_11: the magnitude of the independent noise component    :return: covariance noise matrix    """    delta = 0.5    item = np.exp(-.5 * sqdist / theta_10**2)    return theta_9**2 * item + theta_11**2 * deltadef covariance_function(a, b, hyperparms):    """    covariance function    :param a: the first input vector    :param b: the second input vector    :param hyperparms: hyperparameters for covariance function    :return: kernel matrix    """    # for 1 dimensional data    if a.shape[1] ==1 and b.shape[1] == 1:        sqdist = ((a[:, :, None] - b[:, :, None].T) ** 2).sum(1)        l2_norm = np.sqrt(sqdist)    # for multi-dimensional data    else:        sqdist = (((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))#.T        l2_norm = np.sqrt(sqdist)    kernel = kernel_1(sqdist, hyperparms[0], hyperparms[1]) + \             kernel_2(l2_norm, sqdist, hyperparms[2], hyperparms[3], hyperparms[4]) + \             kernel_3(sqdist, hyperparms[5], hyperparms[6], hyperparms[7]) + \             kernel_4(sqdist, hyperparms[8], hyperparms[9], hyperparms[10])    return kerneldef get_pos_data(info):    """    remove outlier and get positive data    :param info: all information    :return: positive data    """    data = info[:, 1:13].flatten()    neg_indices = np.where(data <= 0)    pos_data = np.delete(data, neg_indices)    return pos_datadef plot_CO2(X_train, y_train):    """    plot CO2 diagram    :param X_train: training data    :param y_train: training targets    :return:    """    num_data = len(y_train)    # plot data    plt.plot(X_train, y_train)def random_sample_test_parms(n_test_hyperparms, train_parms):    """    randomly sample hyperparameters for test    :param n_test_hyperparms: number of test hyperparameters    :param train_parms: training hyperparms, in other words, params_done    :return: test hyperparameters matrix    """    dim_parms = 11    test_parms_matrix = np.zeros(shape=(n_test_hyperparms, dim_parms))    for i in range(dim_parms):        num_gen = n_test_hyperparms + len(train_parms) + 10        test_parms = np.linspace(0.01, 150, num_gen)        ind_done, ind_sample = overlap(train_parms[:,i], test_parms)        test_parms = np.delete(test_parms, ind_sample)        test_parms_sampled = random.sample(test_parms, n_test_hyperparms)        test_parms_matrix[:, i] = np.asarray(test_parms_sampled)    #print test_parms_matrix    return test_parms_matrixdef compute_mar_likelihood(X_train, X_test, y_train, hyperparms):    """    compute log marginal likelihood    :param X_train: training data    :param X_test: test data    :param y_train: training targets    :param hyperparms: hyperparameters    :return: value of log marginal likelihood    """    s = 0.0005  # noise variance and zero mean for noise    N = len(X_train)    n = len(X_test)    K_train = covariance_function(X_train, X_train, hyperparms)    #K_s = covariance_function(X_train, X_test, hyperparms)    #K_ss = covariance_function(X_test, X_test, hyperparms)    L = np.linalg.cholesky(K_train + s * np.eye(N))    m = np.linalg.solve(L, y_train)    alpha = np.linalg.solve(L.T, m)    # compute mean of test points for posterior    #mu_post = np.dot(K_s.T, alpha)    #v = np.linalg.solve(L, K_s)    # compute variance for test points    #var_test = np.diag(K_ss) - np.sum(v ** 2, axis=0)    #stand_devi = np.sqrt(var_test)    # compute log marginal likelihood    log_marg_likelihood = -.5 * np.dot(y_train.T, alpha) - np.diagonal(L).sum(0) - n / 2 * np.log(2 * np.pi)    return log_marg_likelihooddef bayesian_opt(hyperparms_train, hyperparms_test, y_train):    """    Bayesian optimization for computing posterior of training hyperparameters    :param hyperparms_train: training hyperparameters    :param hyperparms_test: test hyperparameters    :param y_train: training targets    :return: posterior means, standard deviation, value of sampled posterior function    """    s = 0.0001  # noise variance and zero mean for noise    n = len(hyperparms_test)  # number of test points    N = len(hyperparms_train)  # number of training points    #print hyperparms_train.shape    num_fun = 1    hyperparms = hyperparms_train[N-1]    K = covariance_function(hyperparms_train, hyperparms_train, hyperparms)    K_s = covariance_function(hyperparms_train, hyperparms_test, hyperparms)    K_ss = covariance_function(hyperparms_test, hyperparms_test, hyperparms)    L = np.linalg.cholesky(K + s * np.eye(N))    m = np.linalg.solve(L, y_train)    alpha = np.linalg.solve(L.T, m)    # compute mean of test points for posterior    mu_post = np.dot(K_s.T, alpha)    v = np.linalg.solve(L, K_s)    # compute variance for test points    var_test = np.diag(K_ss) - np.sum(v ** 2, axis=0)    stand_devi = np.sqrt(var_test)    # sample from test points, in other words, make prediction    L_ = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(v.T, v))    f_post_fun = mu_post.reshape(-1, 1) + np.dot(L_, np.random.normal(size=(n, num_fun)))    return mu_post, stand_devi, f_post_fundef UBC(hyperparms_train, hyperparms_test, mu_post, stand_devi, y):    """    Upper Confidence Bound acquisition function    :param hyperparms_train: training hyperparameters    :param hyperparms_test: test hyperparameters    :param mu_post: posterior means    :param stand_devi: standard deviation    :param y: training targets    :return: next point that need to pick up    """    num_parms = len(hyperparms_train)    kappa = 7    objective = mu_post + kappa * stand_devi    #print objective    indices = np.where(objective == np.max(objective))[0]    indices = np.asarray(indices)    #print indices    next_point = hyperparms_test[indices[0]]    if np.array_equal(hyperparms_train[num_parms - 1], next_point):        return True    return next_pointdef max_LCB(hyperparms_train, hyperparms_test, mu_post, stand_devi, y):    """    figure out the maximum of the Lower Confidence Bound    :param hyperparms_train: training hyperparameters    :param hyperparms_test: test hyperparameters    :param mu_post: posterior means    :param stand_devi: standard deviation    :param y: training targets    :return: next point that need to pick up    """    num_train_parms = len(hyperparms_train)    hyperparms = np.append(hyperparms_train, hyperparms_test, axis=0)    mu = np.append(y, mu_post)    stand_deviation = np.append(np.zeros(num_train_parms), stand_devi)    alpha = 0.5    beta = 4 * np.log(num_train_parms) + 2 * np.log(2 / alpha)    B = np.sqrt(beta) * 1000000    UCB = mu + B * stand_deviation    LCB = mu - B * stand_deviation    print "the maximum of LCB is: "    print np.max(LCB)    print "UCB that below the max(LCB) is: "    print UCB[UCB < np.max(LCB)]    #print hyperparms_testdef TS(hyperparms_train, hyperparms_test, y_train):    """    Thompson Sampling acquisition function    :param hyperparms_train: training hyperparameters    :param hyperparms_test: test hyperparameters    :param y_train: training targets    :return: next point that need to pick up    """    mu_post, stand_devi, f_post_fun = bayesian_opt(hyperparms_train, hyperparms_test, y_train)    max_index = np.where(f_post_fun == np.max(f_post_fun))    next_point = hyperparms_test[max_index[0]]    return next_point.flatten()def acquisition_fun(hyperparms_train, hyperparms_test, mu_post, stand_devi, y):    """    different acquisition functions    :param hyperparms_train: training hyperparameters    :param hyperparms_test: test hyperparameters    :param mu_post: posterior means    :param stand_devi: standard deviation    :param y: training targets    :return: next point that need to pick up    """    next_point = UBC(hyperparms_train, hyperparms_test, mu_post, stand_devi, y)    #next_point = TS(hyperparms_train, hyperparms_test, y)    max_LCB(hyperparms_train, hyperparms_test, mu_post, stand_devi, y)    return next_pointdef tune_hyperparameters_BO(X_train, y_train, X_test):    """    using Bayesian optimization in order to tune hyperparameters    :param X_train: training data    :param y_train: training targets    :param X_test: test data    :return: all values of log marginal likelihoods    """    dim_parms = 11    n_hyperparms_test = 100  # number of test hyperparameters    hyperparms_train = np.zeros(shape=(3, dim_parms)) # define initial hyperparms    for i in range(3):        hyperparms_train[i] = np.random.uniform(0.01, 150, dim_parms) # randomly initial hyperparameters    num_iterations = 1    y_axis = np.zeros(num_iterations)    for k in range(num_iterations):        n_hyperparms_train = len(hyperparms_train)        hyperparms_test = random_sample_test_parms(n_hyperparms_test, hyperparms_train)        log_marg_likelihood = np.zeros(n_hyperparms_train)        for i in range(n_hyperparms_train):            log_marg_likelihood[i] = compute_mar_likelihood(X_train, X_test, y_train, hyperparms_train[i])        # Bayesian optimization for hyperparameters        mu_post, stand_devi, f_post_fun = bayesian_opt(hyperparms_train, hyperparms_test, log_marg_likelihood)        next_point = acquisition_fun(hyperparms_train, hyperparms_test, mu_post, stand_devi, log_marg_likelihood)        hyperparms_train = np.append(hyperparms_train, [next_point], axis=0)        max_index = np.where(log_marg_likelihood == np.max(log_marg_likelihood))[0]        #print max_index        print "**"        print "the " + `k+1` + "th iteration!"        y_axis[k] = np.max(log_marg_likelihood)        print y_axis[k]        #print hyperparms_train[max_index][0]    return y_axisif __name__ == "__main__":    info = np.load('data_list.npy')    # preprocess data    pos_data = get_pos_data(info)    empirical_mean = np.mean(pos_data)    y_train = pos_data - empirical_mean    X_train = (np.arange(len(y_train))).reshape(-1,1)    X_test = (np.arange(np.max(X_train)+1, np.max(X_train)+101)).reshape(-1,1)    # plot CO2    plot_CO2(X_train, y_train)    #plt.show()    # compute covariance function    #kernel = covariance_function(X_train, X_train)    # tune hyperparameters    y_axis = tune_hyperparameters_BO(X_train.reshape(-1,1), y_train, X_test)    x_axis = np.arange(len(y_axis))    print y_axis    # plot convergence    plt.clf()    plt.plot(x_axis, np.abs(y_axis))    #plt.show()