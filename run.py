from micrograd.engine import Value
from micrograd.nn import MLP
from computational_graph_viz import draw_dot

xs = [
    [2.0, 3.0, -1.0], 
    [3.0, -1.0, 0.5], 
    [0.5, 1.0, 1.0], 
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

for k in range(100):
    # forward pass
    ypred = [n(x) for x in xs]
    
    loss = sum([(ygt - yout) **2 for ygt, yout in zip(ys, ypred) ])
    
    # Backward pass
    for p in n.parameters():
        p.grad = 0

    loss.backward()

    for p in n.parameters():
        p.data -= 0.075 * p.grad

    print(k, loss.data)

print(ypred)