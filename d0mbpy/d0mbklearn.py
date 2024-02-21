from typing import Any
import numpy as np
import random

class lin_equation:
    def __init__(self):
        self.w = None
    
    def fit(self,x,y):
        w = (x.T@x)
        w = x.T@y / w
        self.w =  np.array([w])

    def predict(self,x):
        return x.T * self.w
    
    
class Loss:

    def __init__(self,y_hat= 0,y= 0):
        self.y = y
        self.y_hat = y_hat


    @staticmethod
    def mse(y_hat,y):
        return (1/len(y))*(y_hat - y).T@(y_hat - y)
    
    @staticmethod
    def cross_ent(pred,prob):
        return -np.mean(np.sum(np.log(pred+.00001)*prob))





class bayesLinReg:
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y

        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)

        self.x_var = np.var(x)
        self.y_var = np.var(y)
        self.w_mean = 0
        self.w_var = 1

        self.new_mean = None
        self.new_var = None
    
    def post_prior_mean(self):
        new_w_mean = self.w_mean * self.y_var / (self.y_var + len(self.x)*self.w_var)
        sample_mean = (self.x_mean * len(self.x)* self.w_var) / (self.y_var + len(self.x)*self.w_var)

        self.new_mean = new_w_mean + sample_mean
    
    def post_prior_var(self):
        var = ((1/self.w_var) + (len(self.x)/self.y_var))**-1
        self.new_var = var

    def fit(self):
        self.post_prior_mean()
        self.post_prior_var()
    
    def predict(self, x):
        return self.new_mean * x

    def pred_var(self,x):
        return x**2 * self.w_var**2 + self.y_var**2
    
# need to work on this
class bayesClassification(bayesLinReg):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)
    

    @staticmethod
    def sigmoid(x):
        return(np.exp(x))/(1+np.exp(x))

    def predict(self, x):
        return self.sigmoid(super().predict(x))

class kfolds:
    def __init__(self, x, y, model, loss, k_fold):
        self.x = x
        self.y = y
        self.model = model
        self.loss = loss
        self.k_fold = k_fold
        
    @staticmethod
    def split(data,k_fold):
        splits = len(data)//k_fold
        split_data = []
        for i in range(0,len(data),splits):
            split_data.append(data[i:i+splits])
        return split_data
    

    def run(self):
        x = self.split(self.x, self.k_fold)
        y = self.split(self.y, self.k_fold)

        x_copy = x.copy()
        y_copy = y.copy()

        loss = []

        for i in range(self.k_fold):
            val_x = x_copy.pop(i)
            val_y = y_copy.pop(i)

            x_train = np.array(x_copy).flatten()
            y_train = np.array(y_copy).flatten()

            self.model.fit(x_train,y_train)
            pred = np.array(self.model.predict(np.array(val_x)))

            val_y = np.array(val_y)
            
            loss.append(self.loss(pred, val_y))

            x_copy = x.copy()
            y_copy = y.copy()

            self.model.w = None
        return loss




class SVM:
    def __init__(self, lr=.1, iterations=1000) -> None:
        self.iterations = iterations
        self.lr=lr
        self.w = None
        self.b = None

    def fit(self, x,y):

        #x = (x * x +1)**2
        
        
        w = np.random.randn(x.shape[1])
        b = np.random.randn(x.shape[1])

        for i in range(self.iterations):
            pred = x @ w.T + b



            loss = self.hinge_loss(x,y,w,b)
            w_grad = np.sum(self.gradient_w(x,y,w,pred))
            b_grad = np.sum(self.gradient_b(x,y,pred))
            w = w - self.lr * w_grad
            b = b - self.lr * b_grad

            pred = x @ w.T + b

            loss = self.hinge_loss(x,y,w,b)
            print(loss)
        self.w = w
        self.b = b
    
    def predict(self,x):
        #x = (x * x +1)**2
        return x @ self.w.T + self.b

    @staticmethod
    def gradient_w(x,y,w,pred,lama=.1):
        w_grad = np.where(1-y*(pred) >0, lama*w - 1/len(x) * np.sum(y*x),0)
        return w_grad
    
    @staticmethod
    def gradient_b(x,y,pred):
        b_grad = np.where(1-y*(pred) >0, 1/len(x) * np.sum(y),0)
        return b_grad

    @staticmethod
    def hinge_loss(x,y,w,b, lama=.1):
        y_pred = x @ w.T + b
        return 1/len(x) * np.sum(np.where(y-y_pred>0,y-y_pred,0) + lama * np.linalg.norm(w))


class1_x = np.random.randn(20, 1) + np.array([2])  # Shift class 1 points

# Generate points for Class 2
#class2_x = np.random.randn(10, 2) + np.array([-2, -2])  # Shift class 2 points

# Labels
class1_y = np.ones(10)  # Class 1 labels
class2_y = -np.ones(10)  # Class 2 labels

# Combine into one dataset
X = class1_x#np.vstack((class1_x, class2_x))
y = np.concatenate((class1_y, class2_y))

model = SVM()
model.fit(X,y)

print('x',X[2])
print('y',y[2])
print(model.predict(X[2])[0])