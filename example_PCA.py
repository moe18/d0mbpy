import d0mbpy.d0mbpy as dp
import numpy as np
a = np.random.rand(3,3)

x = dp.LinAlg(a.tolist())

# normilize data
x_norm  = (x - x.mean(axis=1)) / x.std(axis=1)

print(x.cov())
print(np.cov(a))
