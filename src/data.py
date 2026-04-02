import numpy as np 

def generate_data(nb_points, noise_std=0.3):
    x = np.linspace(-5, 5, nb_points)
    y_true = np.sin(x) + 0.1*x
    y_noise = y_true + np.random.normal(0, noise_std, size=y_true.shape)
    return x, y_true, y_noise
