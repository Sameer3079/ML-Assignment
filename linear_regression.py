import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])


def calculate_cost(param_1, param_2, x, y):
    cost = np.sum((y - ((x * param_1) + param_2)) ** 2) / (2 * x.__len__())
    return cost


def line_of_best_fit(x, y):
    x_avg = np.average(x)
    y_avg = np.average(y)

    p_1 = np.sum((y - y_avg) * (x - x_avg)) / np.sum((x - x_avg) ** 2)
    p_2 = y_avg - p_1 * x_avg

    return p_1, p_2


def gradient_descent(x, y, param_1, param_2, alpha, iterations):
    m = param_1
    c = param_2

    for iteration in range(iterations):
        d_m = (2 * np.sum((y - ((m * x) + c)) * x)) / x.__len__()

        d_c = (2 * np.sum(y - ((m * x) + c))) / x.__len__()

        # print([d_m, m, c])

        m = m + (alpha * d_m)
        c = c + (alpha * d_c)

    return m, c


print("Initial Cost: ", calculate_cost(0, 0, x, y))

param_1, param_2 = line_of_best_fit(x, y)
print("\nCost of After Applying LSM: ", calculate_cost(param_1, param_2, x, y))
print("Parameters: ", param_1, " ", param_2)

alpha = 0.001
iterations = 10000
p1_start_value, p2_start_value = 0, 0
param_1, param_2 = gradient_descent(x, y, p1_start_value, p2_start_value, alpha, iterations)
print("\nCost of After Applying Gradient Descent: ", calculate_cost(param_1, param_2, x, y))
print("Parameters: ", param_1, " ", param_2)
