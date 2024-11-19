import math

class Value:
    def __init__(self, data:float, _children=(), _op='', label='') -> None:
        self.data :float = data
        self.grad :float = 0
        self._backward = lambda: None
        self._op :str = _op
        self._prev : tuple[Value] = _children
        self.label = label
    
    def backward(self):
        topo: list[Value] = []
        visited = set()

        def build_topo(v: Value):
            if v in visited:
                return
            visited.add(v)
            for child_v in v._prev:
                build_topo(child_v)
            topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+") 

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        return out
    
    def __pow__(self, other:"Value") -> "Value":
        assert isinstance(other, (int, float)), "Only supporting int / float powers"
        out = Value(self.data**other.data, (self, ), f"**{other}")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other:"Value") -> "Value":
        return self * other**-1
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)