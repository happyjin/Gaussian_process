import numpy as np
from scipy.linalg import block_diag

a1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
a2 = np.array([[2,2,2],[2,2,2],[2,2,2]])
a3 = np.array([[3,3,3],[3,3,3],[3,3,3]])

b = np.array([1,2,3]).reshape(-1, 1)
a = np.array([4,5,6,7,8,9,0]).reshape(-1, 1)
num_a = len(a)
num_b = len(b)

print np.tile(a, (1, num_b))
print np.tile(b.T, (num_a, 1))