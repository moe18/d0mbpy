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
            
            loss.append(self.loss.mse(pred, val_y))

            x_copy = x.copy()
            y_copy = y.copy()

            self.model.w = None
        return loss




        

x = np.array([1,2,3,4,5,6,7,8,9,10])


y = np.array([2,4,6,8,10,1,3,5,7,9])
print(kfolds(x,y,lin_equation(),Loss,5).run())

