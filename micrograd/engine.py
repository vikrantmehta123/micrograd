import math
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data:float, _children=(), _op='', label='') -> None:
        self.data :float = data
        self.grad :float = 0
        self._op :str = _op
        self._prev : tuple[Value] = set(_children)
        self.label = label

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+") 
        return out
    
    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data * other.data, (self, other), "*")
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        return Value(t, (self, ), 'tanh')

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
