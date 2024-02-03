import math

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
            self.grad += d0mbTorch(out.grad) * d0mbTorch(other.data)
            other.grad += d0mbTorch(out.grad) * d0mbTorch(self.data)

        out._backward = _backward

        return out
    

    def __pow__(self,other:int|float):
        other = other if isinstance(other, d0mbTorch) else d0mbTorch(other)
        out = self.data ** other.data
        out = d0mbTorch(out, child=(self,other), op='**')

        def _backward():
            self.grad += d0mbTorch(out.grad) * d0mbTorch(other.data) * d0mbTorch(self.data) **d0mbTorch((other.data - 1))

        out._backward = _backward

        return out
    
    def __truediv__(self,other):
        out = self * other**(-1)

        return out

    
    def __neg__(self): return self * -1

    def __rmul__(self, other): return self * other
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return self - other


    def exp(self):
        out = d0mbTorch(math.exp(self.data), child=(self,), op='exp')

        def _backward():
            self.grad += out.grad * out.data
        
        self._backward = _backward

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


x1 = d0mbTorch(2)
x2 = d0mbTorch(4)


v = x1 **3 *x2

v.grad = 1
v.backward()
x1.grad.data = 0.0
x2.grad.data = 0.0

v.backward()
print(x1.grad.child[0])


