import d0mbpy.d0mbpy as dp
import numpy as np
from numpy.linalg import eig

a = np.random.rand(3,3)
#a = np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3]])
x = dp.LinAlg(a)

# normilize data
x_norm  = (x - x.mean(axis=0)) / x.std(axis=0)

x_cov = x_norm.cov()

print(x_cov.eigen_values())

x_norm_np = (a-np.mean(a,axis=0))/np.std(a,axis=0)
print(eig(np.cov(x_norm_np)))
