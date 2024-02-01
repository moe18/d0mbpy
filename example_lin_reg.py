import d0mbpy.d0mbtorch as dt 

x = [2,5,6,3,1]
w1 = dt.d0mbTorch(.2)

x0 = dt.d0mbTorch(1)
y = [4,7,7,2,5]

y_preds = [x1 * w1 + x0 for x1 in x]

mse = dt.d0mbTorch(0)
mse = sum([(yi-y_hat)**2 for y_hat, yi in zip(y_preds, y)])/len(y_preds)

for i in range(1000):
    mse.grad = 1
    mse.backward()

    lr = .01
    new_w1 = w1 - lr * w1.grad
    new_x0 = x0 - lr * x0.grad

    w1.data = new_w1.data
    x0.data = new_x0.data

    w1.grad = 0.0
    x0.grad = 0.0

    y_preds = [x1 * w1 + x0 for x1 in x]

    mse = dt.d0mbTorch(0)
    mse = sum([(yi-y_hat)**2 for y_hat, yi in zip(y_preds, y)])/len(y_preds)
print('w1:',w1.data)
print('x0:',x0.data)
print('mse:',mse.data)

