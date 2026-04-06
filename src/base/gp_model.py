import numpy as np


def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    x1.reshape(-1, 1)
    x2.reshape(-1, 1)

    sqdist = (x1 - x2.T) ** 2

    return variance * np.exp(-0.5 * sqdist / length_scale**2)


def gp_predict(x_train, y_train, x_test, length_scale=1.0, variance=1.0, noise=1e-2):
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    K = rbf_kernel(x_train, x_train, length_scale, variance) + noise * np.eye(len(x_train))
    K_inv = np.linalg.inv(K)

    K_star = rbf_kernel(x_test, x_train, length_scale, variance)

    K_star_star = rbf_kernel(x_test, x_test, length_scale, variance)

    y_pred = K_star @ K_inv @ y_train

    cov = K_star_star - K_star @ K_inv @ K_star.T

    variance = np.diag(cov)
    variance = np.maximum(variance, 0)
    std = np.sqrt(variance)

    return y_pred.flatten(), std.flatten()
