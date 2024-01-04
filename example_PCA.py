import d0mbpy.d0mbpy as dp
import numpy as np
from numpy.linalg import eig

a = np.random.rand(3,3)
x = dp.LinAlg(a)


# normilize data
x_norm  = (x - x.mean(axis=0)) / x.std(axis=0)

# get cov of data
x_cov = x_norm.cov()
# get eigen values and eigen vectors
e_vals, e_vec = x_cov.eigen_values()

n = 2
pca = (dp.LinAlg(e_vec.transpose()[:n])* x_norm.transpose()).transpose()

print(pca)
