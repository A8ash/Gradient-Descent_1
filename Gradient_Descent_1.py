import numpy as np

x = np.array([1, 2, 3, 4, 5, 6])  # hours studied
y = np.array([2.1, 3.9, 5.2, 6.1, 7.8, 9.0])  # test score
w, b, lr = 0.0, 0.0, 0.03  # start + step size
for step in range(1500):
    err = w * x + b - y
    loss = (err ** 2).mean()  # how wrong?
    dw = 2 * (err * x).mean()  # slope along w
    db = 2 * err.mean()  # slope along b (corrected from err ** x)
    w, b = w - lr * dw, b - lr * db  # step downhill

print(f"w = {w:.2f}, b = {b:.2f}, loss = {loss:.3f}")