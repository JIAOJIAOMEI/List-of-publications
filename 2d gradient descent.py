# @Author  : Mei Jiaojiao
# @Time    : 2024/3/15 17:51
# @Software: PyCharm
# @File    : 2d gradient descent.py


import numpy as np
import matplotlib.pyplot as plt

# Define the function to minimize
f = lambda x, y: (x - 3.5) ** 2 + (y - 3.5) ** 2

# Define the gradient function
g_x = lambda x, y: 2 * (x - 3.5)
g_y = lambda x, y: 2 * (y - 3.5)

# Generate x and y values for plotting
x = np.linspace(0, 7, 100)
y = np.linspace(0, 7, 100)

# Compute z values for plotting
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Set the initial guess for the minimum
x_current = 5.5
y_current = 6.3

# Set the learning rate
eta = 0.1

# Perform gradient descent
x_history = [x_current]
y_history = [y_current]
tolerance = 0.0000001
iteration = 0

def gradient_descent(eta, x_current, y_current, x_history, y_history, tolerance, iteration):
    while True:
        x_previous, y_previous = x_current, y_current
        x_current = x_previous - eta * g_x(x_previous, y_previous)
        y_current = y_previous - eta * g_y(x_previous, y_previous)
        x_history.append(x_current)
        y_history.append(y_current)
        iteration += 1
        if np.sqrt((x_current - x_previous) ** 2 + (y_current - y_previous) ** 2) < tolerance:
            break
    return x_current, y_current, x_history, y_history, iteration


plt.figure(figsize=(21, 5))

# Plot contour plot with gradient descent trajectory
for i, eta in enumerate([0.1, 0.5, 0.9], start=1):
    x_current, y_current, x_history, y_history, iteration = gradient_descent(eta, 5.5, 6.3, [5.5], [6.3], 0.0000001, 0)
    plt.subplot(1, 3, i)
    plt.contourf(X, Y, Z, levels=50, cmap='coolwarm')
    plt.colorbar(label='f(x, y)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(x_history, y_history, color='blue', label='Trajectory', s=150)
    plt.scatter(x_history[0], y_history[0], color='red', label='Start', s=150)
    plt.scatter(x_history[-1], y_history[-1], color='green', label='End', s=150)
    plt.plot(x_history, y_history, color='blue', linestyle='--')
    plt.title(f'Learning Rate: {eta}')
    plt.text(2.4, 6.5, f'Iterations: {iteration}', fontsize=12)
    plt.text(2.4, 6, f'Learning rate: {eta}', fontsize=12)
    plt.text(2.4, 5.5, f'Tolerance: {tolerance}', fontsize=12)
    plt.legend(loc='upper left')

plt.savefig('gradient descent 2d.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
