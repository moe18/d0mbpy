
class d0mbTorch:

    def __init__(self, data=None, child=(), op=None) -> None:

        self.data = data
        self.child = child
        self.op = op
        self._backward = lambda: None
        self.grad = 0

    def __add__(self, other):
        other = other if isinstance(other, d0mbTorch) else d0mbTorch(other)

        out = self.data + other.data
        out = d0mbTorch(out, child=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, d0mbTorch) else d0mbTorch(other)

        out = self.data * other.data
        out = d0mbTorch(out, child=(self,other), op='*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out
    

    def __pow__(self,other):
        other = other if isinstance(other, d0mbTorch) else d0mbTorch(other)
        out = self.data ** other.data
        out = d0mbTorch(out, child=(self,other), op='**')

        def _backward():
            self.grad += out.grad * other.data * self.data **(other.data - 1)

        out._backward = _backward

        return out
    

    def backward(self):
        visited = set()
        backset = []

        def topu(v):
            if v not in visited:
                visited.add(v)

                for i in v.child:
                    i._backward()
                    topu(i)
            
            backset.append(v)
        
        topu(self)
        
        for i in reversed(backset):
            i._backward()

            

    
    def __repr__(self) -> str:
        return f"Value:{self.data}, child:{self.child}, grad: {self.grad}, op: {self.op}"


x1 = d0mbTorch(3)
x2 = d0mbTorch(1)
w1 = d0mbTorch(.5)
w2 = d0mbTorch(.1)
x1w1 = x1 * w1
x2w2 = x2 * w2
v1 = d0mbTorch(.5) * x1w1
v2 = d0mbTorch(.5) * x2w2

v = v1 + v2

v.grad = 1
v.backward()
print(x1)