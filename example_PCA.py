import d0mbpy.d0mbpy as dp
import numpy as np

x = dp.LinAlg(np.random.rand(10,3).tolist())

# normilize data
for i in range(x.shape()[1]):
    print(x - x.slice_matrix(i).mean())