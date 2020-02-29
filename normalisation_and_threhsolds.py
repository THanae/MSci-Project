import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mva import obtain_bdt, get_x_and_y


def normalise_b():
    bdt = obtain_bdt()
    X, y = get_x_and_y()
    Xb = pd.read_csv('b_mc_data2.csv')
    Xb = Xb.drop(columns='Unnamed: 0')
    print(len(X) - len(Xb))
    b_br = 10 ** (-7) * 0.5
    acceptance = 0.4
    b_cross_section = 280 * 10 ** (-6)
    b_luminosity = 3 * 10 ** 15
    n_bb = b_cross_section * b_luminosity
    eff = len(Xb)/((506603 + 521404) / 0.14118)
    number_of_b_decays = n_bb * acceptance * b_br * eff
    print(len(Xb), number_of_b_decays)
    normalisation_factor = number_of_b_decays/len(Xb)
    print(f'normalisation factor = {normalisation_factor}')
    # predictions = bdt.decision_function(Xb)
    # a = np.linspace(-1, 1, 2000)
    # to_plot = []
    # for _e in a:
    #     e = sum([1 if i > _e else 0 for i in predictions])/len(Xb)
    #     number_obs = n_bb * acceptance * b_br * e
    #     to_plot.append(number_obs)
    # plt.plot(a, to_plot)
    # # plt.ylabel('number of observations')
    # # plt.xlabel('number of observations')
    # plt.show()
    return normalisation_factor


if __name__ == '__main__':
    normalise_b()
