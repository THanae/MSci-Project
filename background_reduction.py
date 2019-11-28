from data_loader import load_data, add_branches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from masses import masses


def plot_pkmu_mass(data_frame):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    particles = ['Kminus_P', 'proton_P', 'mu1_P']
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    data_frame['sum_m'] = sum_m
    print('final length of the data', len(sum_m))
    # plt.hist(sum_m, bins=50, range=[4500, 6500])
    n, b, p = plt.hist(sum_m, bins=200, range=[1000, 8000])
    plt.vlines(2286, 0, np.max(n)*0.9)  # mass of lambda c
    plt.xlabel('$m_{pK\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    # TODO look at pid outside and inside peak
    return sum_m


def identify_p_k_j_psi(data_frame, to_plot=False):
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    particles = ['mu1_P', 'tauMu_P']
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    print(len(mom_x), len(data_frame))
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    data_frame['dimuon_mass'] = sum_m
    if to_plot:
        plt.hist(sum_m, bins=100, range=[0, 8000])
        plt.vlines(3097, ymin=0, ymax=17000)
        # plt.fill_between([2828, 3317], 0, 17000, color='red', alpha=0.3)
        plt.fill_between([3000, 3150], 0, 17000, color='red', alpha=0.3)
        plt.xlabel('$m_{\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
    compare_data = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3000)]
    print(compare_data['Kminus_PIDe'].describe())
    print(compare_data['proton_PIDe'].describe())
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3150) | (data_frame['dimuon_mass'] < 3000)]
    # data_frame = data_frame.reset_index()
    # print(data_frame['Kminus_PIDe'].describe())
    # print(data_frame['proton_PIDe'].describe())
    # TODO look at pid outside and inside peak
    return data_frame


def reduce_background(data_frame):
    # background reduction code goes here
    # background due to lc
    # background due to d
    # data_frame = data_frame[(data_frame['Kminus_PIDe'] > -8) & (data_frame['Kminus_PIDe'] < -2)]
    # data_frame = data_frame[(data_frame['proton_PIDe'] > -8) & (data_frame['proton_PIDe'] < -2)]
    data_frame = data_frame[data_frame['tauMu_IPCHI2_OWNPV'] > 9]
    data_frame = data_frame[data_frame['proton_IPCHI2_OWNPV'] > 9]
    data_frame = data_frame[data_frame['Kminus_IPCHI2_OWNPV'] > 9]
    data_frame = data_frame[data_frame['mu1_IPCHI2_OWNPV'] > 9]
    data_frame = data_frame[data_frame['mu1_isMuon']]
    data_frame = data_frame.reset_index()
    return data_frame


if __name__ == '__main__':
    a = load_data(add_branches())
    a.dropna(inplace=True)