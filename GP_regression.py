import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=10, suppress=True)

f = lambda x: np.sin(0.9*x).flatten()

# RBF kernel
def RBF_kernel(a,b,kernel_parameter):
    sqdist = np.sum(a**2,1).reshape(-1, 1) - 2*np.dot(a, b.T) + np.sum(b**2,1).reshape(1, -1)
    return np.exp(-.5 * (1 / kernel_parameter) * sqdist)

N = 10     # number of training points
n = 50     # number of test points
s = 0.0005  # noise variance and zero mean for noise
num_fun = 10 # number of prior function
kernel_parameter = 1

# Sample N input points of noisy version of the function evaluated at these points
t = np.linspace(-5, 5, N)
X_train = np.random.uniform(-5, 5, size=(N,1))
y_train = f(X_train) + np.sqrt(s)*np.random.randn(N)
plt.plot(X_train, y_train, 'ro')
#plt.show()

# points we're going to make predictions at.
X_test = np.linspace(-5, 5, n).reshape(-1, 1)
# sampling from a multivariate Gaussian for prior function
mu = np.zeros((n, 1)) # zero mean for multivariate Gaussian
K = RBF_kernel(X_test, X_test, kernel_parameter) # covariance matrix for prior function
B = np.linalg.cholesky(K + 1e-6 * np.eye(n))
f_prior = mu + np.dot(B, np.random.normal(size=(n, num_fun)))
print f_prior.shape

# plot prior function
plt.plot(X_test, f_prior)
plt.axis([-5, 5, -3, 3])
#plt.show()

# make prediction for test data by posterior
K_train = RBF_kernel(X_train, X_train, kernel_parameter)
L = np.linalg.cholesky(K_train + s * np.eye(N))
m = np.linalg.solve(L, y_train)
alpha = np.linalg.solve(L.T, m)
K_s = RBF_kernel(X_train, X_test, kernel_parameter)
mu = np.dot(K_s.T, alpha)
#cov_matrix = RBF_kernel(X_test)
#print mu
K_ss = RBF_kernel(X_test, X_test, kernel_parameter)
v = np.linalg.solve(L, K_s)
cov_matrix = K_ss - np.dot(v.T, v)
# sample from posterior function
L_ = np.linalg.cholesky(cov_matrix + 1e-6 * np.eye(n))
f_post = mu.reshape(-1,1) + np.dot(L_, np.random.normal(size=(n, num_fun)))
plt.figure(1)
plt.clf()
plt.plot(X_test, f_post)

plt.figure(2)
plt.clf()
plt.plot(X_train, y_train, 'r+', ms=20)
plt.plot(X_test, f(X_test), 'b-')
plt.gca().fill_between(X_test.flat, mu-3*s, mu+3*s, color="#dddddd")
plt.plot(X_test, mu, 'r--', lw=2)
#plt.savefig('predictive.png', bbox_inches='tight')
plt.title('Mean predictions plus 3 st.deviations')
plt.axis([-5, 5, -3, 3])

plt.show()