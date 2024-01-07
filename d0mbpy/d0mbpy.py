

def abs(x):
    if x < 0:
        return x * -1
    else:
        return x


class LinAlg:
    def __init__(self, data):
        self.data = data


    def __repr__(self):
        return f"vec({self.data})"
    
    def __pow__(self,expon):
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
        if len(other.data) != len(self.data):
            print('make sure the lengths of your input are equal')
        else:
            vals = []
            hold = []
            shape = self.shape()
            for i in range(shape[0]):
                hold = []
                Q = 0
                for j in range(shape[1]):
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
        R = E * self

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
    

class proba(LinAlg):
    def __init__(self) -> None:
        pass

    @staticmethod
    def expectaion(vals, proba=None):
        if proba:
            return vals * proba.transpose()
        else:
            _sum = 0
            count = 0
            for val in vals:
                _sum+=val
                count +=1
            return _sum/count
             
           
    def var(self, vals, proba):
        mean = self.expectaion(vals,proba)
        diff = vals - mean
        var = (diff)**2
        var = self.expectaion(var)
        return var


a = LinAlg([[1,2,3]])
b = LinAlg([[.5,0,.5]])

c = proba()

print(c.var(a,b))