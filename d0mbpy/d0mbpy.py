

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


    def __add__(self, other):
        if len(other.data) != len(self.data):
            print('make sure the lengths of your input are equal')
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
        if min(other.shape()) == 1:
            for i in range(len(self.data)):
                Q = 0
                for j in range(len(other.data)):
                    Q+= self.data[i][j] * other.data[j][0]
                vals.append([Q])
        else:    
            for i in range(len(self.data)):
                hold = []
                for j in range(len(self.data)):
                    Q = 0
                    for k in range(len(self.data)): 
                        Q+= self.data[i][k] * other.data[k][j]
                    hold.append(Q)
                vals.append(hold)
        return LinAlg(vals)
    

    def __truediv__(self, other):
        vals = []
        if min(other.shape()) == 1:
            for i in range(len(self.data)):
                Q = 0
                for j in range(len(other.data)):
                    Q+= self.data[i][j] / other.data[j][0]
                vals.append([Q])
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
        print('means=', means)
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


    
