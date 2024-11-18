import math
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data:float, _children=(), _op='', label='') -> None:
        self.data :float = data
        self.grad :float = 0
        self._op :str = _op
        self._prev : tuple[Value] = set(_children)

    def __add__(self, other : float | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+") 
        return out
    
    def __mul__(self, other : float | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data * other.data, (self, other), "*")
        return out
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
