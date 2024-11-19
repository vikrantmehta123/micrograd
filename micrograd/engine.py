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
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

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
