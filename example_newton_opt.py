import d0mbpy.d0mbpy as dp


def f(x):
    return (x[0] - x[1])**2

# if this wors correctly it should go from 1,2 to 1,1
x = dp.LinAlg([[1,2]])
for i in range(10):
    
    H = x.hessian(f).invert()
    g = x.jacobian(f)

    x = x - (H * g).flatten()
    print(x)