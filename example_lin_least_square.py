from d0mbpy import d0mbpy as dp

def mse(x,w,y):
    return dp.LinAlg([[.5]])* (w*x - y)**2


x = dp.LinAlg([[1,2],
               [3,4]])

w = dp.LinAlg([[.5,.5]])

y = dp.LinAlg([[1,2]])

lr = dp.LinAlg([[.01]])
for i in range(1000):
    grad = w.transpose()*(w*x - y)
    grad = grad.flatten()
    x = x - lr*grad
    print(mse(x,w,y))

