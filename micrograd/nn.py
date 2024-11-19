import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x:list[Value]):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    

class Layer:
    def __init__(self, nin, nout) -> None:
        # nin -> Number of inputs to the neuron.
        # nout -> Number of neurons in the layer
        self.neurons = [Neuron(nin=nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs