import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data, add_branches
from masses import masses
from plotting_functions import plot_compare_data


def plot_pkmu_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    particles = ['Kminus_P', 'proton_P', 'mu1_P']
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    data_frame['sum_m'] = sum_m
    print('final length of the data', len(sum_m))
    compare_data = data_frame[(data_frame['sum_m'] < 2300) & (data_frame['sum_m'] > 2200)]
    background_selection = data_frame[(data_frame['sum_m'] > 2300) | (data_frame['sum_m'] < 2200)]
    if to_plot:
        # plt.hist(sum_m, bins=50, range=[4500, 6500])
        n, b, p = plt.hist(sum_m, bins=70, range=[1500, 5000])
        plt.vlines(masses['Lc'], 0, np.max(n) * 0.9)  # mass of lambda c
        plt.fill_between([2190, 2310], 0, np.max(n), color='red', alpha=0.3)  # [2828, 3317]
        plt.xlabel('$m_{pK\\mu}$')
        plt.ylabel('occurrences')
        plt.show()

        selection_range = 100
        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['proton_PIDp', ['proton_PIDp', 'proton_PIDK'],
                                           ['proton_PIDp', 'proton_PIDmu']], signal_name='lambda_0')
        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['Kminus_PIDK', ['Kminus_PIDK', 'Kminus_PIDp'],
                                           ['Kminus_PIDK', 'Kminus_PIDmu']], signal_name='lambda_0')
        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['mu1_PIDmu', ['mu1_PIDmu', 'mu1_PIDp'],
                                           ['mu1_PIDmu', 'mu1_PIDK']], signal_name='lambda_0')
        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['tauMu_PIDmu', ['tauMu_PIDmu', 'tauMu_PIDp'],
                                           ['tauMu_PIDmu', 'tauMu_PIDK']], signal_name='lambda_0')

    data_frame = data_frame[data_frame['sum_m'] > 2300]  # remove lower part too or not?
    # TODO look at pid outside and inside peak
    # can remove events by removing events with pkmu mass smaller than some number (eg 2400)
    return data_frame


def plot_kmu_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu']]
    particles = ['Kminus_P', 'mu1_P']
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    data_frame['sum_m'] = sum_m
    print('final length of the data', len(sum_m))
    if to_plot:
        # plt.hist(sum_m, bins=50, range=[4500, 6500])
        n, b, p = plt.hist(sum_m, bins=100, range=[0, 4000])
        plt.axvline(masses['D0'], c='k')  # mass of d0
        plt.fill_between([1830, 1880], 0, np.max(n), color='red', alpha=0.3)  # [2828, 3317]
        plt.xlabel('$m_{K\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
    # need to find way to get muon with opposite charge to kaon
    # then can remove events with kmu mass smaller than upper limit for peak
    # data_frame = data_frame[(data_frame['sum_m'] <1840) | (data_frame['sum_m'] > 1880)]
    data_frame = data_frame[(data_frame['sum_m'] > 1880)]
    return data_frame


def plot_mumu_mass(data_frame):
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    particles = ['mu1_P', 'tauMu_P']
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    print(len(mom_x), len(data_frame))
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    data_frame['dimuon_mass'] = sum_m
    plt.hist(sum_m, bins=100, range=[0, 8000])
    # plt.fill_between([2828, 3317], 0, 17000, color='red', alpha=0.3)
    plt.xlabel('$m_{\\mu\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    return data_frame


def identify_p_k_j_psi(data_frame, to_plot=True):
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    particles = ['mu1_P', 'tauMu_P']
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    data_frame['dimuon_mass'] = sum_m
    plt.hist2d(data_frame['Lb_M'], data_frame['dimuon_mass'], bins=30, range=[[2000, 8000], [300, 4000]])
    plt.axvline(masses['Lb'])
    plt.axhline(masses['J/psi'])
    plt.colorbar()
    plt.show()
    compare_data = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3050)]
    # compare_data=data_frame
    compare_data = compare_data[(compare_data['Lb_M'] < 5650) & (compare_data['Lb_M'] > 5590)]
    # plt.hist(compare_data['dimuon_mass'], bins=50, range=[2000, 8000])
    # plt.show()
    # n, b, p = plt.hist(compare_data['Lb_M'], bins=100, range=[4000, 7000])
    # plt.fill_between([5590, 5650], 0, np.max(n)*1.1, color='red', alpha=0.3)
    # plt.axvline(masses['Lb'], c='k')
    # plt.show()
    print('comp', len(compare_data))
    data_frame = data_frame[(data_frame['dimuon_mass'] > 3150) | (data_frame['dimuon_mass'] < 3050)]
    # data_frame = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3050)]
    data_frame = data_frame[(data_frame['dimuon_mass'] > 3700) | (data_frame['dimuon_mass'] < 3650)]
    # background_selection = data_frame[(data_frame['Lb_M'] > 5800) | (data_frame['Lb_M'] < 5200)]
    background_selection = data_frame[(data_frame['Lb_M'] > 5800)]
    # background_selection = data_frame[(data_frame['dimuon_mass'] > 3050) | (data_frame['dimuon_mass'] < 3150)]
    print('back', len(background_selection))
    if to_plot:
        n, b, t = plt.hist(sum_m, bins=100, range=[0, 5000])
        plt.vlines(3097, ymin=0, ymax=np.max(n))
        plt.vlines(3686, 0, np.max(n))
        plt.fill_between([3040, 3160], 0, np.max(n), color='red', alpha=0.3)  # [2828, 3317]
        plt.fill_between([3640, 3710], 0, np.max(n), color='red', alpha=0.3)  # [2828, 3317]
        plt.xlabel('$m_{\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()

        # fig, axs = plt.subplots(1, 2)
        # axs[0].hist(compare_data['Lb_pmu_ISOLATION_BDT1'], bins=30, density=True)
        # axs[0].set_title('Lb_pmu_ISOLATION_BDT1 jpsi')
        # axs[1].hist(background_selection['Lb_pmu_ISOLATION_BDT1'], bins=30, density=True)
        # axs[1].set_title('Lb_pmu_ISOLATION_BDT1 background')
        plt.hist(compare_data['Lb_pmu_ISOLATION_BDT1'], bins=30, density=True)
        plt.hist(background_selection['Lb_pmu_ISOLATION_BDT1'], bins=30, density=True, alpha=0.3)
        plt.show()

        selection_range = 100
        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['proton_PIDp', ['proton_PIDp', 'proton_PIDK'],
                                           ['proton_PIDp', 'proton_PIDmu']], signal_name='jpsi')
        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['Kminus_PIDK', ['Kminus_PIDK', 'Kminus_PIDp'],
                                           ['Kminus_PIDK', 'Kminus_PIDmu']], signal_name='jpsi')
        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['mu1_PIDmu', ['mu1_PIDmu', 'mu1_PIDp'],
                                           ['mu1_PIDmu', 'mu1_PIDK']], signal_name='jpsi')
        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['tauMu_PIDmu', ['tauMu_PIDmu', 'tauMu_PIDp'],
                                           ['tauMu_PIDmu', 'tauMu_PIDK']], signal_name='jpsi')

        plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                          columns_to_plot=['tauMu_IPCHI2_OWNPV', 'mu1_IPCHI2_OWNPV', 'proton_IPCHI2_OWNPV',
                                           'Kminus_IPCHI2_OWNPV'], signal_name='jpsi')

    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3150) | (data_frame['dimuon_mass'] < 3000)]
    # data_frame = data_frame.reset_index()
    # print(data_frame['Kminus_PIDe'].describe())
    # print(data_frame['proton_PIDe'].describe())
    # TODO look at pid outside and inside peak
    # need to look fro jpsi in the lambda b mass region
    # TODO need to make cut around the dimuon mass to remove jpsi and psi2s events if true (make arg)
    # look at Lb_pmu_ISOLATION_BDT1

    return data_frame


def clean_cuts(data_frame, to_plot=False):
    # data_frame = data_frame[data_frame['mu1_L0Global_Dec']]
    # data_frame = data_frame[data_frame['tauMu_L0Global_Dec']]
    if to_plot:
        columns_to_plot = ['proton_P', 'proton_PT', 'Kminus_P', 'Kminus_PT', 'mu1_P', 'mu1_PT', 'tauMu_P', 'tauMu_PT']
        fig, axs = plt.subplots(4, 2, gridspec_kw={'hspace': 0.5}, figsize=(10, 6))
        for i in range(len(columns_to_plot)):
            name = columns_to_plot[i]
            range_end = 35e3 if 'T' not in name else 5500
            column = 0 if i % 2 == 0 else 1
            axs[np.int(np.floor(i / 2)), column].hist(data_frame[name], 100, range=[0, range_end])
            axs[np.int(np.floor(i / 2)), column].set_title(name)
        plt.show()
        columns_to_plot = ['proton_PIDp', 'Kminus_PIDK', 'mu1_PIDmu', 'tauMu_PIDmu', 'proton_PIDK', 'proton_PIDmu',
                           'Kminus_PIDp', 'Kminus_PIDmu', 'mu1_PIDp', 'mu1_PIDK', 'tauMu_PIDp', 'tauMu_PIDK']
        fig, axs = plt.subplots(6, 2, gridspec_kw={'hspace': 0.6}, figsize=(10, 10))
        for i in range(len(columns_to_plot)):
            name = columns_to_plot[i]
            range_end = 50
            column = 0 if i % 2 == 0 else 1
            axs[np.int(np.floor(i / 2)), column].hist(data_frame[name], 100, range=[0, range_end])
            axs[np.int(np.floor(i / 2)), column].set_title(name)
        plt.show()

        fig, axs = plt.subplots(4, 2, gridspec_kw={'hspace': 0.5}, figsize=(10, 6))
        axs[0, 0].hist(data_frame['proton_PIDp'] - data_frame['proton_PIDK'], 100, range=[0, 100])
        axs[0, 0].set_title('proton_PIDp-proton_PIDK')
        axs[0, 1].hist(data_frame['proton_PIDp'] - data_frame['proton_PIDmu'], 100, range=[0, 100])
        axs[0, 1].set_title('proton_PIDp-proton_PIDmu')
        axs[1, 0].hist(data_frame['Kminus_PIDK'] - data_frame['Kminus_PIDp'], 100, range=[0, 100])
        axs[1, 0].set_title('Kminus_PIDK-Kminus_PIDp')
        axs[1, 1].hist(data_frame['Kminus_PIDK'] - data_frame['Kminus_PIDmu'], 100, range=[0, 100])
        axs[1, 1].set_title('Kminus_PIDK-Kminus_PIDmu')
        axs[2, 0].hist(data_frame['mu1_PIDmu'] - data_frame['mu1_PIDp'], 100, range=[0, 100])
        axs[2, 0].set_title('mu1_PIDmu-mu1_PIDp')
        axs[2, 1].hist(data_frame['mu1_PIDmu'] - data_frame['mu1_PIDK'], 100, range=[0, 100])
        axs[2, 1].set_title('mu1_PIDmu-mu1_PIDK')
        axs[3, 0].hist(data_frame['tauMu_PIDmu'] - data_frame['tauMu_PIDp'], 100, range=[0, 100])
        axs[3, 0].set_title('tauMu_PIDmu-tauMu_PIDp')
        axs[3, 1].hist(data_frame['tauMu_PIDmu'] - data_frame['tauMu_PIDK'], 100, range=[0, 100])
        axs[3, 1].set_title('tauMu_PIDmu-tauMu_PIDK')
        plt.show()

    proton_P_threshold = 15e3
    proton_PT_threshold = 1000
    mu1_P_threshold = 10e3
    mu1_PT_threshold = 1500
    tauMu_P_threshold = 10e3
    tauMu_PT_threshold = 1500
    proton_PIDp_threshold = 10
    proton_PIDpK_threshold = 10

    plt.hist2d(data_frame['mu1_PT'], data_frame['tauMu_PT'], bins=50, range=[[0, 8e3], [0, 8e3]])
    plt.show()

    data_frame = data_frame[data_frame['proton_P'] > proton_P_threshold]
    data_frame = data_frame[data_frame['proton_PT'] > proton_PT_threshold]
    data_frame = data_frame[data_frame['mu1_P'] > mu1_P_threshold]
    data_frame = data_frame[data_frame['mu1_PT'] > mu1_PT_threshold]
    data_frame = data_frame[data_frame['tauMu_P'] > tauMu_P_threshold]
    data_frame = data_frame[data_frame['tauMu_PT'] > tauMu_PT_threshold]
    data_frame = data_frame[data_frame['proton_PIDp'] > proton_PIDp_threshold]
    data_frame = data_frame[data_frame['proton_PIDp'] - data_frame['proton_PIDK'] > proton_PIDpK_threshold]
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def chi2_cleaning(data_frame):
    # TODO compare with control data
    plt.hist(data_frame['Lb_FD_OWNPV'], bins=50, range=[0, 100])
    plt.show()
    data_frame = data_frame[data_frame['pKmu_ENDVERTEX_CHI2'] < 9]
    data_frame = data_frame[data_frame['Lb_FD_OWNPV'] < 16]
    return data_frame


def impact_parameter_cleaning(data_frame: pd.DataFrame, threshold: int = 9):
    """
    Ensure that the particles do not originate from the primary vertex
    :param data_frame: data
    :param threshold: threshold number below which data is removed
    :return: cleaned data frame
    """
    data_frame = data_frame[data_frame['tauMu_IPCHI2_OWNPV'] > threshold]
    data_frame = data_frame[data_frame['proton_IPCHI2_OWNPV'] > 16]
    data_frame = data_frame[data_frame['Kminus_IPCHI2_OWNPV'] > threshold]
    data_frame = data_frame[data_frame['mu1_IPCHI2_OWNPV'] > threshold]
    data_frame = data_frame.reset_index()
    return data_frame


def pid_cleaning(data_frame):
    """
    Uses the pid of other known decays to estimate the pid of the particles in the pkmutau decays
    For j/psi, we know the pid of actual protons and kaons and can infer the pid for the pkmutau decay from there
    :param data_frame: data
    :return: cleaned data
    """
    # need to do estimate for lambda c
    data_frame = data_frame[data_frame['proton_PIDp'] > 15]
    data_frame = data_frame[data_frame['Kminus_PIDK'] - data_frame['Kminus_PIDp'] > 15]  # or 9?
    data_frame = data_frame[data_frame['mu1_PIDmu'] - data_frame['mu1_PIDK'] > 15]  # 9 or not?
    return data_frame


def reduce_background(data_frame):
    data_frame = clean_cuts(data_frame)
    data_frame = impact_parameter_cleaning(data_frame)
    data_frame = pid_cleaning(data_frame)
    data_frame = chi2_cleaning(data_frame)
    data_frame = data_frame[data_frame['mu1_isMuon']]
    data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT1'] < 0.2]
    data_frame = identify_p_k_j_psi(data_frame, False)
    data_frame = plot_pkmu_mass(data_frame, False)
    # data_frame = plot_kmu_mass(data_frame, False)
    data_frame = data_frame.reset_index()
    return data_frame


if __name__ == '__main__':
    a = load_data(add_branches())
    a.dropna(inplace=True)
    a = clean_cuts(a, True)
    # a = a[a['Lb_pmu_ISOLATION_BDT1'] < 0.2]
    a = a.reset_index()
    df = identify_p_k_j_psi(a, to_plot=True)
    # df = plot_kmu_mass(a, True)
    # df = plot_pkmu_mass(a, True)
    identify_p_k_j_psi(a, to_plot=True)
