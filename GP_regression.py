import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

f = lambda x: np.sin(0.9*x).flatten()

# RBF kernel
def RBF_kernel(a,b,kernel_parameter):
    sqdist = np.sum(a**2,1).reshape(-1, 1) - 2*np.dot(a, b.T) + np.sum(b**2,1).reshape(1, -1)
    dis_mean = 1
    return dis_mean * np.exp(-.5 * (1 / kernel_parameter) * sqdist)

N = 5           # number of training points
n = 50          # number of test points
s = 0.0005      # noise variance and zero mean for noise
num_fun = 10    # number of prior function
kernel_parameter = 1
kernel_stand_deiv = 1 # standard deviation for kernel

# Sample N input points of noisy version of the function evaluated at these points
t = np.linspace(-5, 5, N)
X_train = np.random.uniform(-5, 5, size=(N,1))
y_train = f(X_train) + np.sqrt(s)*np.random.randn(N)
plt.plot(X_train, y_train, 'ro')

# points we're going to make predictions at.
X_test = np.linspace(-5, 5, n).reshape(-1, 1)
# sampling from a multivariate Gaussian for prior function
mu_prior = np.zeros((n, 1)) # zero mean for multivariate Gaussian
K = RBF_kernel(X_test, X_test, kernel_parameter) # covariance matrix for prior function
B = np.linalg.cholesky(K + 1e-6 * np.eye(n))
f_prior = mu_prior + np.dot(B, np.random.normal(size=(n, num_fun)))

# plot prior function
#plt.figure(1)
plt.subplot(2,2,1)
#plt.clf()
plt.fill_between(X_test.flat, 0 - kernel_stand_deiv, 0 + kernel_stand_deiv, color="#dddddd")
plt.plot(X_test, f_prior)
plt.title('samples from the GP prior')
plt.axis([-5, 5, -3, 3])

# make prediction for test data by posterior
K_train = RBF_kernel(X_train, X_train, kernel_parameter)
L = np.linalg.cholesky(K_train + s * np.eye(N))
m = np.linalg.solve(L, y_train)
alpha = np.linalg.solve(L.T, m)
K_s = RBF_kernel(X_train, X_test, kernel_parameter)
# compute mean of test points for posterior
mu_post = np.dot(K_s.T, alpha)
K_ss = RBF_kernel(X_test, X_test, kernel_parameter)
v = np.linalg.solve(L, K_s)

# compute variance for test points
var_test = np.diag(K_ss) - np.sum(v**2, axis=0)
stand_devi = np.sqrt(var_test)

# sample from test points
L_ = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(v.T, v))
f_post = mu_post.reshape(-1, 1) + np.dot(L_, np.random.normal(size=(n, num_fun)))
plt.axis([-5, 5, -3, 3])

#plt.figure(2)
plt.subplot(2,2,2)
#plt.clf()
plt.gca().fill_between(X_test.flat, mu_post - 3 * stand_devi, mu_post + 3 * stand_devi, color="#dddddd")
plt.plot(X_test, f_post)
plt.plot(X_test, mu_post, 'r--', lw=2)
plt.title('samples from the GP posterior')
plt.axis([-5, 5, -3, 3])

#plt.figure(3)
plt.subplot(2,2,3)
#plt.clf()
plt.plot(X_train, y_train, 'r+', ms=20)
plt.plot(X_test, f(X_test), 'b-')
plt.gca().fill_between(X_test.flat, mu_post - 3 * stand_devi, mu_post + 3 * stand_devi, color="#dddddd")
plt.plot(X_test, mu_post, 'r--', lw=2)
#plt.savefig('predictive.png', bbox_inches='tight')
plt.title('Mean predictions plus 3 st.deviations')
plt.axis([-5, 5, -3, 3])

plt.show()