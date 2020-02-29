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


def f_o_m():
    bdt = obtain_bdt()
    Xb = pd.read_csv('b_mc_data2.csv')
    Xb = Xb.drop(columns='Unnamed: 0')
    data = pd.read_csv('b_mc_data2.csv')
    data = data.drop(columns='Unnamed: 0')
    predictions_signal = bdt.decision_function(Xb)
    predictions_background = bdt.decision_function(data)
    thresholds = np.linspace(-1, 1, 2000)
    signal, background = [], []
    punzi = []
    for t in thresholds:
        s = sum([1 if i >= t else 0 for i in predictions_signal]) * 0.0023072060793360356
        b = sum([1 if i < t else 0 for i in predictions_background])
        signal.append(s)
        background.append(b)
        print(s, b)
        # b = sum([1 if i > t else 0 for i in predictions_background])
        punzi.append(s/np.sqrt(s+b))
    plt.plot(thresholds, [1/i for i in punzi])
    # plt.plot(thresholds, punzi)
    plt.show()
    plt.plot(background, [1/i for i in punzi])
    # plt.plot(background, punzi)
    plt.xlabel('background')
    plt.ylabel('signal significance')
    plt.show()


if __name__ == '__main__':
    f_o_m()
    normalise_b()
