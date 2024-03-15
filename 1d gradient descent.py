# @Author  : Mei Jiaojiao
# @Time    : 2024/3/15 17:49
# @Software: PyCharm
# @File    : 1d gradient descent.py

import numpy as np
import matplotlib.pyplot as plt

# Define the function to minimize
f = lambda x: (x - 3.5) ** 2 - 4.5 * x + 10

# Define the gradient function
g = lambda x: 2 * (x - 3.5) - 4.5

# Generate x values for plotting
x = np.linspace(0, 11.5, 100)

# Compute y values for plotting
y = f(x)

# Plot the function
plt.plot(x, y, label='f(x)')

# Set the initial guess for the minimum
x_min = 5.75

# Set the learning rate
eta = 0.01

# Perform gradient descent
x_current = 10
x_history = [x_current]
tolerance = 0.0000001
iteration = 0


def gradient_descent(eta, x_current, x_history, tolerance, iteration):
    while True:
        x_previous = x_current
        x_current = x_previous - eta * g(x_previous)
        x_history.append(x_current)
        iteration += 1
        if np.abs(x_current - x_previous) < tolerance:
            break
    return x_current, x_history, iteration


plt.figure(figsize=(21, 5))

# Plot the function and gradient descent trajectory
for i, eta in enumerate([0.01, 0.1, 0.9], start=1):
    x_current, x_history, iteration = gradient_descent(eta, 10, [10], 0.0000001, 0)
    plt.subplot(1, 3, i)
    plt.plot(x, y, label='f(x)')
    plt.scatter(x_history, f(np.array(x_history)), color='blue', label='Trajectory', s=150)
    plt.plot(x_history, f(np.array(x_history)), color='blue', linestyle='--')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Learning Rate: {eta}')
    plt.text(0.6, 20, f'Iterations: {iteration}', fontsize=12)
    plt.text(0.6, 18, f'Learning rate: {eta}', fontsize=12)
    plt.text(0.6, 16, f'Final x: {x_current}', fontsize=12)
    plt.scatter(x_history[0], f(np.array(x_history[0])), color='red', label='Start', s=150)
    plt.scatter(x_history[-1], f(np.array(x_history[-1])), color='green', label='End', s=150)
    plt.legend(loc='upper right')

plt.savefig('gradient descent.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
