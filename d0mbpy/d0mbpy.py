
from typing import List

def abs(x: int| float) -> int|float: return x if x>0 else -1 * x

def ln(x: int|float, itterations: int=100)->int|float: return sum((-1)**(i + 1) * ((x-1)**i) / i for i in range(1,itterations))

class LinAlg:
    def __init__(self, data:List[List[int|float]]):
        self.data = data


    def __repr__(self):
        return f"vec({self.data})"
    
    def __pow__(self,expon: int|float):
        vals = []
        hold = []
        shape = self.shape()
        for i in range(shape[0]):
            hold = []
            Q = 0
            for j in range(shape[1]):
                Q = self.data[i][j]**expon
                hold.append(Q)
            vals.append(hold)
        return LinAlg(vals)


    def __add__(self, other):
        if other.shape()[0] == 1 and other.shape()[1]==1:
            vals = []
            for i in range(self.shape()[0]):
                hold = []
                Q = 0
                for j in range(self.shape()[1]):
                    Q = self.data[i][j] + other.data[0][0]
                    hold.append(Q)
                vals.append(hold)
            return LinAlg(vals)
        
        elif (other.shape()[0] == 1):
            vals = []
            for i in range(self.shape()[0]):
                hold = []
                Q = 0
                for j in range(other.shape()[1]):
                    Q = self.data[i][j] + other.data[0][j]
                    hold.append(Q)
                vals.append(hold)
            return LinAlg(vals)
        else:
            vals = []
            for i in range(len(self.data)):
                hold = []
                Q = 0
                for j in range(len(other.data)):
                    Q = self.data[i][j] + other.data[i][j]
                    hold.append(Q)
                vals.append(hold)
            return LinAlg(vals)
        


    def __sub__(self, other):
        if other.shape()[0] == 1 and other.shape()[1]==1:
            vals = []
            for i in range(self.shape()[0]):
                hold = []
                Q = 0
                for j in range(self.shape()[1]):
                    Q = self.data[i][j] - other.data[0][0]
                    hold.append(Q)
                vals.append(hold)
            return LinAlg(vals)
        
        elif (other.shape()[0] == 1):
            vals = []
            for i in range(self.shape()[0]):
                hold = []
                Q = 0
                for j in range(other.shape()[1]):
                    Q = self.data[i][j] - other.data[0][j]
                    hold.append(Q)
                vals.append(hold)
            return LinAlg(vals)
        else:
            vals = []
            for i in range(len(self.data)):
                hold = []
                Q = 0
                for j in range(len(other.data)):
                    Q = self.data[i][j] - other.data[i][j]
                    hold.append(Q)
                vals.append(hold)
            return LinAlg(vals)
        

    def __getitem__(self, index):
        if str(index) == ':':
            print('hi')
        return self.data[index]


    def __mul__(self,other):
        vals = []
        if other.shape()[0] == 1 and other.shape()[1] == 1:
            for i in range(self.shape()[0]):
                hold = []
                for j in range(self.shape()[1]):
                    hold.append(other.data[0][0] * self.data[i][j])
                vals.append(hold)
            return LinAlg(vals)
                
        if min(other.shape()) == 1:
            for i in range(len(self.data)):
                Q = 0
                for j in range(len(other.data)):
                    Q+= self.data[i][j] * other.data[j][0]
                vals.append([Q])
        else:    
            for i in range(len(self.data)):
                hold = []
                for j in range(other.shape()[1]):
                    Q = 0
                    for k in range(len(self.data)): 
                        Q+= self.data[i][k] * other.data[k][j]
                    hold.append(Q)
                vals.append(hold)
        return LinAlg(vals)
    

    def __truediv__(self, other):
        vals = []
        if other.shape()[0] == 1 and other.shape()[1]==1:
            for i in range(len(self.data)):
                Q = 0
                for j in range(len(other.data)):
                    Q+= self.data[i][j] / other.data[j][0]
                vals.append([Q])
        elif (other.shape()[0] == 1):
            vals = []
            for i in range(self.shape()[0]):
                hold = []
                Q = 0
                for j in range(other.shape()[1]):
                    Q = self.data[i][j] / other.data[0][j]
                    hold.append(Q)
                vals.append(hold)
            return LinAlg(vals)

        else:    
            for i in range(len(self.data)):
                hold = []
                for j in range(len(self.data)):
                    Q = 0
                    for k in range(len(self.data)): 
                        Q+= self.data[i][k] / other.data[k][j]
                    hold.append(Q)
                vals.append(hold)
        return LinAlg(vals)


    def shape(self):
        return (len(self.data), len(self.data[0][:]))
        

    def transpose(self):
        vals = []
        if self.shape()[0] == 1: 
            for i in range(max(self.shape())):
                vals.append([self.data[0][i]])
            
        else:
            for i in range(self.shape()[1]):
                hold = []
                for j in range(self.shape()[0]):
                    hold.append(self.data[j][i])
                vals.append(hold)
        #self.data = vals
        return LinAlg(vals)
    

    def vec_size(self):
        vec_l = 0
        for i in self.data[0]:
            vec_l+= abs(i)**self.shape()[1]
        return (vec_l)**(1/self.shape()[1])
    
    @staticmethod
    def l2_norm(vec):
        vec_1 = 0
        for i in vec:
            vec_1+= i**2

        return vec_1**.5


    def l1_norm(self):
        vec_1 = 0
        for i in self.data[0]:
            vec_1+= abs(i)

        return vec_1
    
    def max_norm(self):
        return max(self.data[0])
    
    
    def matrix_norm(self):
        # also known as Frobenius norm
        norm = 0
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                norm += self[i][j]**2
        return (norm)**(1/2)

    def get_diag(self):
        vals = []
        for i in range(self.shape()[0]):
            vals.append(self.data[i][i])
        return vals
    

    def trace(self):
        return sum(self.get_diag())
    
    def slice_matrix(self,col_num):
        return LinAlg([row[col_num] for row in self.data])
        

    def mean(self, axis=0):
        if axis == 0:
            self = self.transpose()
            vals = []
            for i in range(self.shape()[0]):
                s = 0
                count = 0
                for j in range(self.shape()[1]):
                    s+=self.data[i][j]
                    count+=1
                vals.append(s/ (count))
            
            return LinAlg([vals])
        elif axis == 1:
            vals = []
            for i in range(self.shape()[0]):
                s = 0
                count = 0
                for j in range(self.shape()[1]):
                    s+=self.data[i][j]
                    count+=1
                vals.append(s/ (count))
            
            return LinAlg([vals])
    
    def std(self, axis=0):
        xl = self.mean(axis=axis)
        if axis == 0:
            self = self.transpose()
            vals = []
            for i in range(self.shape()[0]):
                s = 0
                count = 0
                for j in range(self.shape()[1]):
                    s+=(self.data[i][j] - xl[0][i])**2
                    count+=1
                vals.append((s/ count)**.5 )
            
            return LinAlg([vals])
        elif axis == 1:
            vals = []
            for i in range(self.shape()[0]):
                s = 0
                count = 0
                for j in range(self.shape()[1]):
                    s+=(self.data[i][j] - xl[0][i])**2
                    count+=1
                vals.append((s/ count)**.5 )
            
            return LinAlg([vals])
        

    def cov(self):
        means = self.mean(axis=1)[0]
        vals = []
        for i in range(self.shape()[0]):
            hold = []
            for j in range(self.shape()[0]):
                cov_sum = 0
                for k in range(self.shape()[1]):
                    x = self.data[i][k]
                    x_val = x - means[i]

                    y = self.data[j][k]
                    y_val = y - means[j]

                    cov_sum+= x_val*y_val
                hold.append(cov_sum/(self.shape()[1]-1))
            vals.append(hold)
        return LinAlg(vals)
    

    def __iter__(self):
        self.i = 0
        self.j = 0
        self.finish = self.shape()[0] * self.shape()[1]
        return self

    def __next__(self):
        
            if self.j >= self.shape()[1]:
                self.i+=1
                if self.i >= self.shape()[0]:
                    raise StopIteration
                self.j = 1
                return self.data[self.i][self.j-1]
            else:
                self.j +=1
                return self.data[self.i][self.j-1]


    @staticmethod
    def zeros(shape):
        vals = []
        for i in range(shape[0]):
            hold = []
            for j in range(shape[1]):
                hold.append(0)
            vals.append(hold)
        return LinAlg(vals)
    

    def right(self):
        vals = []
        for i in range(self.shape()[0]):
            hold = []
            for j in range(self.shape()[1]):
                if i <= j:
                    hold.append(self[i][j])
                else:
                    hold.append(0)
            vals.append(hold)
        return LinAlg(vals)


    def __neg__(self):
        return self * -1

    def qr_dec(self):
        
        shape = self.shape()

        e_vals = []
        for i in range(shape[0]):
            if len(e_vals):
                u = LinAlg([self.data[i]])
                v = LinAlg([self.data[i]])
                for e in e_vals:
                    e = LinAlg([e])
                    u -= e*(v*e.transpose())
            else:
                u = LinAlg([self.data[i]])
            
            norm =self.l2_norm(u)
            val = u/LinAlg([[norm]*shape[0]])
            e_vals.append(val.data[0])

        E = LinAlg(e_vals)
        Q = E.transpose()
        R = Q * self

        return Q, R.right()
    

    def eigen_values(self):
        Q,R = self.qr_dec()
        Q_s = Q
        A = R * Q

        for i in range(1000):
            Q,R = A.qr_dec()
            A = R*Q
            Q_s *= Q
        

        return LinAlg([A.get_diag()]), Q_s
    
    
    def get_eigen_vector(self, A, eigenvalues):
        eigenvectors = []
        I = self.identity(self.shape())
        for lambda_ in eigenvalues:
            # Solve (A - lambda*I)x = 0
            eig_vec = A - LinAlg([[lambda_]])*I
            eig_vec2 = eig_vec.eigen_values()
            eigenvectors.append(eig_vec2.data[0])
        return LinAlg(eigenvectors)
    
    @staticmethod
    def identity(shape):
        vals = []
        for i in range(shape[0]):
            hold = []
            for j in range(shape[1]):
                if i == j:
                    hold.append(1)
                else:
                    hold.append(0)
            vals.append(hold)
        return LinAlg(vals)
    # add random matrix 

    def reshape(self,shape):
        vals = []
        count = 0
        for i in range(shape[0]):
            hold = []
            for j in range(shape[1]):
                hold.append(self[0][count])
                count +=1
            vals.append(hold)
        
        return LinAlg(vals)


    @staticmethod
    def deriv(f, x, eps=.0001, interms=0):
        vals = x.copy()
        vals[interms] = vals[interms]-eps
        return round((f(x) - f(vals)) / eps,2)
    

    @staticmethod
    def second_deriv(f, x, eps=.0001, interms=0):
        val_plus = x.copy()
        val_minus = x.copy()

        val_plus[interms] = val_plus[interms] + eps
        val_minus[interms] = val_minus[interms] - eps


        return round((f(val_plus) - 2*f(x) + f(val_minus)) / eps**2,2)
    
    @staticmethod
    def partial_deriv(f,x, eps=.0001, interms_i=0, interms_j=1):

        a = (f([x[interms_i]+eps, x[interms_j]+eps]) - 
         f([x[interms_i]+eps, x[interms_j]-eps]) -
         f([x[interms_i]-eps, x[interms_j]+eps]) + 
         f([x[interms_i]-eps, x[interms_j]-eps]))
        b = 4 * eps**2

        return round(a/b, 2)

    
    
    def jacobian(self,f):
        vals = []
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                vals.append(self.deriv(f,self.data[i],interms=j))
        return LinAlg([vals]).reshape(self.shape())
    

    
    def second_derive_matrix(self,f):
        vals = []
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                vals.append(self.second_deriv(f,self.data[i],interms=j))
        return LinAlg([vals]).reshape(self.shape())
            
    
    # only works for 2d
    def hessian(self, f):
        vals = []
        for i in range(self.shape()[1]):
            for j in range(self.shape()[1]):
                if i == j:
                    vals.append(self.second_deriv(f,self.data[0],interms=j))
                elif i < j:
                    vals.append(self.partial_deriv(f,self.data[0],interms_i=i, interms_j=j))
                else:
                    vals.append(self.partial_deriv(f,self.data[0],interms_i=j, interms_j=i))
        return LinAlg([vals]).reshape((self.shape()[1],self.shape()[1]))
    
    

    def invert(self):
        Q,R = self.qr_dec()
        
        inv_r = self.zeros(self.shape())

        for i in range(R.shape()[0]):
            inv_r[i][i] = 1/R[i][i]
        for i in range(R.shape()[0]):
            for j in range(R.shape()[1]):
                if i == j:
                    inv_r[i][j] = inv_r[i][j]
                elif i > j:
                    inv_r[i][j] = 0
                
                else:
                    #something is wrong here
                    inv_r[i][j] = -(1/ R[i][i]) *sum(R[i][k] * inv_r[k][j] for k in range(i,j+1))
        return inv_r * Q.transpose()
    
    def flatten(self):
        vals = []
        for i in self:
            vals.append(i)
        return LinAlg([vals])




class proba(LinAlg):
    def __init__(self, data=None, prob=.5) -> None:
        self.data = data
        self.prob = prob
    
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32
        self.state = 42

        self.pi = 3.14159
        self.e = 2.71828

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    @staticmethod
    def expectaion(vals, proba=None):
        if proba:
            return vals * proba.transpose()
        else:
            _sum = 0
            count = 0
            for val in vals[0]:
                _sum+=val
                count +=1
            return LinAlg([[_sum/count]])
             
           
    def var(self):
        mean = self.expectaion(self.data)
        diff = self - mean
        var = (diff)**2
        var = self.expectaion(var)
        return var
    
    def bern_rand(self):
        r = self.rand()
        if r > self.prob:
            return 1
        else:
            return 0

    def bern_dist(self):
        p = self.prob
        q = 1 - self.prob
        e = self.prob
        var = p*q
        dist = {'p':p,
                'q':q,
                'ex':e,
                'var':var}

        return dist
    
    def norm(self,x,mean,var):
        p1 = (1/(2*self.pi*var))**.5
        p2 = self.e**(-((x-mean)**2)/(2*var))
        return p1*p2
    
    def rand_norm(self,mean=0,var=1):
        run = True
        while run:
            r1 = 2 * self.rand() - 1
            r2 = 2 * self.rand() - 1
            s = r1**2 + r2**2

            if 0 < s < 1:
                z0 = r1 * ((-2*ln(s))/s)**.5
                run = False
        return z0+mean * var**.5

    def sigmoid(self, x):
        return 1 / (1+self.e**(-x))




        
