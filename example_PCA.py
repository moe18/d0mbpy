import d0mbpy.d0mbpy as dp
import numpy as np
from numpy.linalg import eig

a = np.random.rand(3,5)
#a = np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3]])
x = dp.LinAlg(a)
x = dp.LinAlg([[1,1,0],
            [1,0,1],
            [0,1,1]])
#x = dp.LinAlg(a)


# normilize data
x_norm  = (x - x.mean(axis=0)) / x.std(axis=0)

x_cov = x_norm.cov()

x_cov = dp.LinAlg([[2.92,.86,-1.15],[.86,6.51,3.32],[-1.15,3.32,4.57]])
e_vals, e_vec = x_cov.eigen_values()
print(e_vals, e_vec.transpose())

