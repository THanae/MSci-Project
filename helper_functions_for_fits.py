import numpy as np
import scipy.special as special


def line(x, m, b):
    return m * x + b


def gaussian(x, mu, sigma):
    f = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return f


def exponential(x, a, b):
    return a * np.exp(x * b)


def exp_gaussian(x, l, mu, sigma):
    f = l / 2 * np.exp(l / 2 * (2 * mu + l * sigma ** 2 - 2 * x))
    complementary_error_function = special.erfc((mu + l * sigma ** 2 - x) / (np.sqrt(2) * sigma))
    return f * complementary_error_function


def lorentzian(x, x0, gamma):
    return 1 / np.pi * (0.5 * gamma / ((x - x0) ** 2 + (0.5 * gamma) ** 2))
