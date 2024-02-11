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
    
    @staticmethod
    def mse(y_hat,y):
        return (1/len(y))*(y_hat - y).T@(y_hat - y)


class kfolds:
    def __init__(self, data, model, loss, k_fold):
        self.data = data
        self.model = model
        self.loss = loss
        self.k_fold = k_fold
        

    def split(self):
        splits = len(self.data)//self.k_fold
        split_data = []
        for i in range(0,len(self.data),splits):
            split_data.append(self.data[i:i+splits])
        return split_data
    

    # need to create a way to pass x and y into spliter and fit the model correctly
    def run(self):
        folds = self.split()

        for i in range(self.k_fold):
            self.model.fit()
            for j in 

        


x = np.array([1,2,3,4,5,6,7,8,9,10])

split(x,5)
y = np.array([2,4,5,7,8])


model   = lin_equation()
model.fit(x,y)

pred = model.predict(x)

print(model.mse(pred,y))

