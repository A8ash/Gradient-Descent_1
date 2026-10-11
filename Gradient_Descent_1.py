import numpy as np
import matplotlib.pyplot as plt

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

# Plot the original data points
plt.scatter(x, y, color='blue', label='Actual Data (Points)', zorder=5)

# Generate points for the regression line
x_line = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
y_line = w * x_line + b

# Plot the regression line
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression Line (y = {w:.2f}x + {b:.2f})')

# Add labels, title and legend
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Hours Studied vs Test Score with Regression Line')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.show()