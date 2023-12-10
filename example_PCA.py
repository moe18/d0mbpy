import d0mbpy.d0mbpy as dp
import numpy as np

x = dp.LinAlg(np.random.rand(10,3).tolist())

# normilize data
x_norm  = (x - x.mean(axis=1)) / x.std(axis=1)

print(x_norm)
