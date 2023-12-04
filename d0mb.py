

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
    def __getitem__(self, index):
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

    def shape(self):
        return (len(self.data), len(self.data[0][:]))
        
    def transpose(self):
        vals = []
        if self.shape()[0] == 1: 
            for i in range(max(self.shape())):
                vals.append([self.data[0][i]])
            
        else:
            for i in range(self.shape()[0]):
                hold = []
                for j in range(self.shape()[1]):
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
