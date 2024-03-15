# @Author  : Mei Jiaojiao
# @Time    : 2024/3/13 20:33
# @Software: PyCharm
# @File    : functionplot.py

import matplotlib.pyplot as plt
import numpy as np


def F1(x):
    result = []
    for i in x:
        k = (-1) * i * np.sin(np.sqrt(np.abs(i)))
        result.append(k)
    return sum(result)


def F2(x):
    result = [np.power(i, 2) - (10 * np.cos(2 * np.pi * i)) + 10 for i in x]
    return sum(result)


def F3(x):
    dim = len(x)
    part1 = sum([np.power(i, 2) for i in x])
    part2 = sum([np.cos(2 * np.pi * i) for i in x])
    result = (-20) * np.exp(-0.2 * np.sqrt((1 / dim) * part1)) + (-np.exp((1 / dim) * part2) + 20 + np.e)
    return result


# Create a meshgrid of x and y values
x = np.linspace(-500, 500, 100)
y = np.linspace(-500, 500, 100)
X, Y = np.meshgrid(x, y)
Z1 = F1([X, Y])


x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z2 = F2([X, Y])

x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)
Z3 = F3([X, Y])

# Create a new figure and set its size
fig = plt.figure(figsize=(15, 6))
# white background
fig.patch.set_facecolor('white')

# Plot the functions in 3D subplots
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap="coolwarm")
# plot projection on xy plane
ax1.contour(X, Y, Z1, zdir='z', offset=-900, cmap="coolwarm")
ax1.set_title('F1')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap="coolwarm")
# plot projection on xy plane
ax2.contour(X, Y, Z2, zdir='z', offset=0, cmap="coolwarm")
ax2.set_title('F2')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z3, cmap="coolwarm")
# plot projection on xy plane
ax3.contour(X, Y, Z3, zdir='z', offset=0, cmap="coolwarm")
ax3.set_title('F3')

# Add x-, y-, and z-labels to all subplots, grid off, axis off
for ax in fig.axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

# Adjust the spacing between subplots to avoid overlap
plt.tight_layout()

plt.savefig("2d functions.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
# Show the figure
plt.show()