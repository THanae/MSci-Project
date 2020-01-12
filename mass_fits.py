import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
from helper_functions_for_fits import line, exp_gaussian, gaussian, lorentzian


def lb_mc_mass():
    lb_mass = np.load('Lb_MC_mass.npy')
    _range = [5550, 5700]
    _bins = 100
    starting_point = [5620, 15]
    x = np.linspace(_range[0], _range[1], _bins)
    n, b, p = plt.hist(lb_mass, bins=_bins, range=_range, density=True, label='pkmumu MC data')
    print(spo.curve_fit(gaussian, ((b[1:] + b[:-1]) / 2), n, p0=starting_point))
    print(spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.5, 5620, 15]))
    (d, e, f), _ = spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.5, 5620, 15])
    (a, b), pcov = spo.curve_fit(gaussian, ((b[1:] + b[:-1]) / 2), n, p0=starting_point)
    print(np.sqrt(np.diag(pcov)))  # one std error of params
    plt.plot(x, gaussian(x, a, b), label='gaussian')
    plt.plot(x, exp_gaussian(x, d, e, f), label='exp gaussian')
    plt.ylabel('Rate of occurrence (normalised)')
    plt.xlabel('Mass')
    plt.legend()
    plt.show()


def b_mc_mass():
    b_mass = np.load('B_MC_mass.npy')
    # _range = [4000, 6500]
    # _range = [5000, 5750]
    _range = [4000, 8000]
    _bins = 100
    x = np.linspace(_range[0], _range[1], _bins)
    starting_point = [5279, 500]
    n, b, p = plt.hist(b_mass, bins=_bins, range=_range, density=True, label='B MC data')
    plt.plot(x, 0.71*gaussian(x, 5321, 206))
    plt.plot(x, 0.69*exp_gaussian(x, 5.11e-3, 5176, 137))
    print(spo.curve_fit(gaussian, ((b[1:] + b[:-1]) / 2), n, p0=starting_point))
    print(spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.5, 5279, 15]))
    print(spo.curve_fit(lorentzian, ((b[1:] + b[:-1]) / 2), n, p0=[5279, 15]))
    (d, e, f), _ = spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.5, 5279, 15])
    (g, h), _ = spo.curve_fit(lorentzian, ((b[1:] + b[:-1]) / 2), n, p0=[5279, 1])
    (a, b), pcov = spo.curve_fit(gaussian, ((b[1:] + b[:-1]) / 2), n, p0=starting_point)
    print(np.sqrt(np.diag(pcov)))  # one std error of params
    # plt.plot(x, gaussian(x, a, b), label='gaussian')
    # plt.plot(x, exp_gaussian(x, d, e, f), label='exp gaussian')
    # plt.plot(x, lorentzian(x, g, h), label='lorentzian')

    plt.ylabel('Rate of occurrence (normalised)')
    plt.xlabel('Mass')
    plt.legend()
    plt.show()


def pkmutau_background():
    pkmutau_mass = np.load('pkmutau_mass_no_bdt_pid.npy')
    # _range = [4000, 15000]
    _range = [8000, 15000]
    _bins = np.int((_range[1] - _range[0])/200)
    x = np.linspace(8000, 15000, _bins)
    starting_point = [0.15, 15]
    n, b, p = plt.hist(pkmutau_mass, bins=_bins, range=_range, label='pkmutau data')
    plt.plot(x, line(x, -4.4e-03, 66))
    print(spo.curve_fit(line, ((b[1:] + b[:-1]) / 2), n, p0=starting_point))
    (a, b), pcov = spo.curve_fit(line, ((b[1:] + b[:-1]) / 2), n, p0=starting_point)
    # print(np.sqrt(np.diag(pcov)))  # one std error of params
    # plt.plot(x, line(x, a, b), label='line')
    plt.ylabel('Rate of occurrence')
    plt.xlabel('Mass')
    plt.legend()
    plt.show()
    pkmutau_mass_cleaned = np.load('pkmutau_mass_cleaned.npy')
    _bins = np.int((_range[1] - _range[0]) / 200)
    n, b, p = plt.hist(pkmutau_mass_cleaned, bins=_bins, range=_range, label='pkmutau data (cleaned)')
    plt.plot(x, line(x, -4.4e-03/4, 66/4))
    # (a, b), pcov = spo.curve_fit(line, ((b[1:] + b[:-1]) / 2), n, p0=starting_point)
    # print(a, b, -4.4e-03/4, 66/4)
    # print(np.sqrt(np.diag(pcov)))  # one std error of params
    # plt.plot(x, line(x, a, b), label='line')
    plt.show()


def fit_real_mass_plots():
    pkmumu_mass = np.load('pkmumu_mass_cleaned.npy')
    pkmutau_mass = np.load('pkmutau_mass_cleaned.npy')
    _range = [4500, 8000]
    _range = [4000, 15000]
    _bins = np.int((_range[1] - _range[0]) / 100)
    x = np.linspace(_range[0], _range[1], _bins)
    n, b, p = plt.hist(pkmutau_mass, bins=_bins, range=_range, label='pkmutau data (cleaned)')
    area = sum(np.diff(b)*n)
    plt.plot(x, area*gaussian(x, 5884, 740))
    # x = np.linspace(8000, _range[1], _bins)
    # plt.plot(x, line(x, -4.4e-03 / 4, 66 / 4)/2)
    # (a, b), pcov = spo.curve_fit(gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[5620, 206])
    # print(a, b)
    plt.ylabel('Rate of occurrence')
    plt.xlabel('Mass')
    plt.legend()
    plt.show()
    # _range = [5500, 5700]
    _range = [4000, 7000]
    _bins = np.int((_range[1] - _range[0]) / 20)
    x = np.linspace(_range[0], _range[1], _bins)
    n, b, p = plt.hist(pkmumu_mass, bins=_bins, range=_range, density=True, label='pkmumu data (cleaned)')
    area = sum(np.diff(b) * n)
    plt.plot(x, area * 0.1 * gaussian(x, 5602, 44))
    # (a, b), pcov = spo.curve_fit(gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[5620, 16])
    # print(np.sqrt(np.diag(pcov)))
    # print(a, b)
    plt.show()


if __name__ == '__main__':
    pkmutau_background()
    fit_real_mass_plots()
