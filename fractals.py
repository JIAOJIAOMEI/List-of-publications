# @Author  : Mei Jiaojiao
# @Time    : 2024/3/13 18:05
# @Software: PyCharm
# @File    : fractals.py

import matplotlib.pyplot as plt
import numpy as np
import datetime

root_list = {}


def f(x):
    return (x ** 3 - 1) / (3 * x ** 2)


root_list['f'] = [1, -1 / 2 + 1j * np.sqrt(3) / 2, -1 / 2 - 1j * np.sqrt(3) / 2]


def id_root(z_list, root_list):
    find_goal = 1.e-10 * np.ones(len(z_list))
    root_id = -1 * np.ones(len(z_list))
    for r in root_list:
        root_id = np.where(np.abs(z_list - r * np.ones(len(z_list))) < find_goal,
                           np.ones(len(z_list)) * root_list.index(r), root_id)
    return root_id


interval_left = -1.2
interval_right = 1.2
interval_down = -1.2
interval_up = 1.2

num_x = 1000
num_y = 1000

precision_goal = 1.e-10
max_iter = 50

print('Started computation at ' + str(datetime.datetime.now()))

x_vals = np.linspace(interval_left, interval_right, num=num_x)
y_vals = np.linspace(interval_down, interval_up, num=num_y)


def plot_newton_fractal(func, perform_shading=False):
    global z1_list
    z_list = []
    for x in x_vals:
        for y in y_vals:
            z_list.append(x + 1j * y)

    res_list = np.array(z_list)
    rel_diff = np.ones(len(res_list))
    counter = np.zeros(len(res_list)).astype(int)
    overall_counter = 0
    prec_goal_list = np.ones(len(res_list)) * precision_goal

    while np.any(rel_diff) > precision_goal and overall_counter < max_iter:
        diff = eval(func + '(res_list)')
        z1_list = res_list - diff
        rel_diff = np.abs(diff / res_list)
        res_list = z1_list
        counter = counter + np.greater(rel_diff, prec_goal_list)
        overall_counter += 1

    n_root = id_root(z1_list, root_list[func]).astype(int)

    if perform_shading:
        n_root = n_root - 0.4 * np.log(counter / np.max(counter))

    n_root_contour = np.transpose(np.reshape(n_root, (num_x, num_y)))

    print('Finished computation at ' + str(datetime.datetime.now()))

    plt.figure()
    plt.xlabel("$Re(z)$", fontsize=12)
    plt.ylabel("$Im(z)$", fontsize=12)
    plt.imshow(n_root_contour, extent=[interval_left, interval_right, interval_down, interval_up], cmap='rainbow')
    plt.axis('on')
    plt.title('Newton Fractal')
    plt.text(0.5, 1.08,
             'Grid: ' + str(num_x) + 'x' + str(num_y) + ', Tolerance: ' + str(precision_goal) + ', Iterations: ' + str(
                 max_iter),
             horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.savefig('newton-fractal.png', bbox_inches='tight', dpi=300)
    plt.close()

    print('Finished creating matshow plot at ' + str(datetime.datetime.now()))


plot_newton_fractal('f', perform_shading=True)

print('Finished computation and plotting at ' + str(datetime.datetime.now()))
