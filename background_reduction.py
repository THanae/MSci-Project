import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from data_loader import load_data, add_branches
from masses import masses, get_mass
from plotting_functions import plot_compare_data, plot_columns


def plot_pkmu_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pkmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    compare_data = data_frame[(data_frame['pkmu_mass'] < 2300) & (data_frame['pkmu_mass'] > 2200)]
    background_selection = data_frame[(data_frame['pkmu_mass'] > 2300) | (data_frame['pkmu_mass'] < 2200)]
    if to_plot:
        # plt.hist(sum_m, bins=50, range=[4500, 6500])
        n, b, p = plt.hist(data_frame['pkmu_mass'], bins=70, range=[1500, 5000])
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

    data_frame = data_frame[data_frame['pkmu_mass'] > 2300]  # remove lower part too or not?
    # TODO look at pid outside and inside peak
    # can remove events by removing events with pkmu mass smaller than some number (eg 2400)
    return data_frame


def plot_kmu_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['tauMu_P', 'mu']]
    data_frame['kmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)

    mu_peak = data_frame[(data_frame['dimuon_mass'] > 1620) & (data_frame['dimuon_mass'] < 1840)]
    mu_peak = mu_peak[mu_peak['mu1_ID'] > 0]
    data_frame.loc[data_frame['mu1_ID'] < 0, 'kmu_mass'] = 25000
    print(len(mu_peak))
    print(data_frame['mu1_ID'].describe())
    if to_plot:
        # plt.hist(sum_m, bins=50, range=[4500, 6500])
        n, b, p = plt.hist(data_frame['kmu_mass'], bins=100, range=[0, 4000])
        plt.axvline(masses['D0'], c='k')  # mass of d0
        plt.axvline(masses['rho0'], c='k')  # mass of rho0
        plt.axvline(masses['rho1450'], c='k')  # mass of rho0
        plt.fill_between([1830, 1880], 0, np.max(n), color='red', alpha=0.3)  # [2828, 3317]
        plt.xlabel('$m_{K\\mu}$')
        plt.ylabel('occurrences')
        plt.show()

        n, b, p = plt.hist(mu_peak['kmu_mass'], bins=100, range=[500, 3500])
        # plt.axvline(masses['D0'], c='k')  # mass of d0
        plt.axvline(masses['kstar'], c='k')
        plt.xlabel('$m_{K\\mu}$')
        plt.ylabel('occurrences')
        plt.title('Kmu mass for the dimuon peak')
        plt.show()

        n, b, p = plt.hist(data_frame['dimuon_mass'], bins=100, range=[0, 4000])
        plt.axvline(masses['D0'], c='k')  # mass of d0
        plt.xlabel('$m_{\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.title('Dimuon mass')
        plt.show()

        plt.hist2d(mu_peak['kmu_mass'], mu_peak['dimuon_mass'], bins=20, norm=LogNorm())
        plt.axvline(masses['kstar'], c='k')
        plt.xlabel('$m_{K\\mu}$')
        plt.ylabel('$m_{\\mu\\mu}$')
        plt.colorbar()
        plt.axhline(masses['Lb'])
        plt.show()
    return data_frame


def plot_mumu_mass(data_frame):
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['dimuon_mass'] = sum_m
    plt.hist(sum_m, bins=100, range=[0, 8000])
    # plt.fill_between([2828, 3317], 0, 17000, color='red', alpha=0.3)
    plt.xlabel('$m_{\\mu\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    return data_frame


def identify_p_k_j_psi(data_frame, to_plot=True):
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['dimuon_mass'] = sum_m
    compare_data = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3050)]
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
        plt.hist2d(data_frame['Lb_M'], data_frame['dimuon_mass'], bins=30, range=[[2000, 8000], [300, 4000]],
                   norm=LogNorm())
        plt.axvline(masses['Lb'])
        plt.axhline(masses['J/psi'])
        plt.xlabel('$m_{pK\\mu\\mu}$')
        plt.ylabel('$m_{\\mu\\mu}$')
        plt.colorbar()
        plt.show()

        n, b, t = plt.hist(sum_m, bins=100, range=[0, 5000])
        plt.vlines(3097, ymin=0, ymax=np.max(n))
        plt.vlines(3686, 0, np.max(n))
        plt.fill_between([3040, 3160], 0, np.max(n), color='red', alpha=0.3)  # [2828, 3317]
        plt.fill_between([3640, 3710], 0, np.max(n), color='red', alpha=0.3)  # [2828, 3317]
        plt.xlabel('$m_{\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()

        n, b, t = plt.hist(compare_data['Lb_FD_OWNPV'], bins=100, range=[0, 100])
        plt.title('Lb flight distance for the jpsi data')
        plt.ylabel('occurrences')
        plt.show()

        plt.hist(compare_data['Lb_pmu_ISOLATION_BDT1'], bins=30, density=True)
        plt.hist(background_selection['Lb_pmu_ISOLATION_BDT1'], bins=30, density=True, alpha=0.3)
        plt.show()
        plt.hist(compare_data['Lb_FD_OWNPV'], bins=80, density=True, range=[0, 80])
        plt.hist(background_selection['Lb_FD_OWNPV'], bins=80, density=True, alpha=0.3, range=[0, 80])
        plt.show()

        mu_peak = data_frame[(data_frame['dimuon_mass'] > 1760) & (data_frame['dimuon_mass'] < 1860)]
        mu_broader = data_frame[(data_frame['dimuon_mass'] > 1500) & (data_frame['dimuon_mass'] < 2000)]
        mu_low = data_frame[(data_frame['dimuon_mass'] < 1500)]
        fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 0.5}, figsize=(10, 6))
        axs[0].hist2d(compare_data['mu1_PIDmu'], compare_data['tauMu_PIDmu'], bins=50, range=[[0, 15], [0, 15]],
                      norm=LogNorm())
        axs[0].set_xlabel('mu1_PIDmu')
        axs[0].set_ylabel('tauMu_PIDmu')
        axs[0].set_title('JPSI')

        axs[1].hist2d(mu_peak['mu1_PIDmu'], mu_peak['tauMu_PIDmu'], bins=50, range=[[0, 15], [0, 15]], norm=LogNorm())
        axs[1].set_xlabel('mu1_PIDmu')
        axs[1].set_ylabel('tauMu_PIDmu')
        axs[1].set_title('Background')
        # axs[1].colorbar()
        plt.show()

        fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 0.5}, figsize=(10, 6))
        axs[0].hist2d(mu_broader['mu1_PIDmu'], mu_broader['tauMu_PIDmu'], bins=50, range=[[0, 15], [0, 15]],
                      norm=LogNorm())
        axs[0].set_xlabel('mu1_PIDmu')
        axs[0].set_ylabel('tauMu_PIDmu')
        axs[0].set_title('Wider part of the peak')

        axs[1].hist2d(mu_low['mu1_PIDmu'], mu_low['tauMu_PIDmu'], bins=50, range=[[0, 15], [0, 15]], norm=LogNorm())
        axs[1].set_xlabel('mu1_PIDmu')
        axs[1].set_ylabel('tauMu_PIDmu')
        axs[1].set_title('Dimuon mass below 1500 MeV')
        # axs[1].colorbar()
        plt.show()

        fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 0.5}, figsize=(10, 6))
        axs[0].hist2d(compare_data['mu1_PIDK'], compare_data['tauMu_PIDK'], bins=50, range=[[0, 15], [0, 15]],
                      norm=LogNorm())
        axs[0].set_xlabel('mu1_PIDK')
        axs[0].set_ylabel('tauMu_PIDK')
        axs[0].set_title('JPSI')

        axs[1].hist2d(mu_peak['mu1_PIDK'], mu_peak['tauMu_PIDK'], bins=50, range=[[0, 15], [0, 15]], norm=LogNorm())
        axs[1].set_xlabel('mu1_PIDK')
        axs[1].set_ylabel('tauMu_PIDK')
        axs[1].set_title('Background')
        # axs[1].colorbar()
        plt.show()

        plot_compare_data(compare_data, mu_peak, histogram_range=15, columns_to_plot=['mu1_PIDmu', 'tauMu_PIDmu'],
                          signal_name='jpsi')

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
        plot_compare_data(compare_data, background_selection, histogram_range=2000,
                          columns_to_plot=['Lb_FD_OWNPV', 'pKmu_ENDVERTEX_CHI2', 'Lb_FDCHI2_OWNPV'], signal_name='jpsi')
    return data_frame
    # return compare_data


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

        plot_columns(data_frame=data_frame, histogram_range=[0, 100], bins=100,
                     columns_to_plot=[['proton_PIDp', 'proton_PIDK'], ['proton_PIDp', 'proton_PIDmu'],
                                      ['Kminus_PIDK', 'Kminus_PIDp'], ['Kminus_PIDK', 'Kminus_PIDmu'],
                                      ['mu1_PIDmu', 'mu1_PIDp'], ['mu1_PIDmu', 'mu1_PIDK'],
                                      ['tauMu_PIDmu', 'tauMu_PIDp'], ['tauMu_PIDmu', 'tauMu_PIDK']])

        plt.hist2d(data_frame['mu1_PT'], data_frame['tauMu_PT'], bins=50, range=[[0, 8e3], [0, 8e3]], norm=LogNorm())
        plt.xlabel('mu1_PT')
        plt.ylabel('tauMu_PT')
        plt.colorbar()
        plt.show()

        plt.hist2d(data_frame['mu1_PIDmu'], data_frame['tauMu_PIDmu'], bins=50, range=[[0, 15], [0, 15]],
                   norm=LogNorm())
        plt.xlabel('mu1_PIDmu')
        plt.ylabel('tauMu_PIDmu')
        plt.colorbar()
        plt.show()

    proton_P_threshold = 15e3
    proton_PT_threshold = 1000
    mu1_P_threshold = 10e3
    mu1_PT_threshold = 1500
    tauMu_P_threshold = 10e3
    tauMu_PT_threshold = 1500
    proton_PIDp_threshold = 10
    proton_PIDpK_threshold = 10



    data_frame = data_frame[data_frame['proton_P'] > proton_P_threshold]
    data_frame = data_frame[data_frame['proton_PT'] > proton_PT_threshold]
    data_frame = data_frame[data_frame['mu1_P'] > mu1_P_threshold]
    data_frame = data_frame[data_frame['mu1_PT'] > mu1_PT_threshold]
    data_frame = data_frame[data_frame['tauMu_P'] > tauMu_P_threshold]
    data_frame = data_frame[data_frame['tauMu_PT'] > tauMu_PT_threshold]
    data_frame = data_frame[data_frame['proton_PIDp'] > proton_PIDp_threshold]
    data_frame = data_frame[data_frame['proton_PIDp'] - data_frame['proton_PIDK'] > proton_PIDpK_threshold]

    # data_frame = data_frame[(data_frame['tauMu_PIDmu'] > 3) & (data_frame['tauMu_PIDmu'] < 9)]
    # data_frame = data_frame[(data_frame['mu1_PIDmu'] > 3) & (data_frame['mu1_PIDmu'] < 9)]
    data_frame = data_frame[(data_frame['tauMu_PIDmu'] > 5)]
    data_frame = data_frame[(data_frame['mu1_PIDmu'] > 5)]

    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def chi2_cleaning(data_frame):
    # TODO compare with control data
    plt.hist(data_frame['Lb_FD_OWNPV'], bins=50, range=[0, 100])
    plt.title('Lb_FD_OWNPV')
    plt.show()
    data_frame = data_frame[data_frame['pKmu_ENDVERTEX_CHI2'] < 9]
    data_frame = data_frame[data_frame['Lb_FDCHI2_OWNPV'] > 300]
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
    data_frame = data_frame[data_frame['Kminus_PIDK'] - data_frame['Kminus_PIDp'] > 10]  # or 9?
    data_frame = data_frame[data_frame['Kminus_PIDK'] - data_frame['Kminus_PIDmu'] > 10]  # or 9?
    data_frame = data_frame[data_frame['Kminus_PIDK'] > 10]  # or 9?
    data_frame = data_frame[data_frame['proton_PIDp'] > 15]  # or 9?
    data_frame = data_frame[data_frame['proton_PIDp'] - data_frame['proton_PIDK'] > 15]  # or 9?
    data_frame = data_frame[data_frame['mu1_PIDmu'] - data_frame['mu1_PIDK'] > 15]  # 9 or not?
    return data_frame


def transverse_momentum_cleaning(data_frame, to_plot: bool = False):
    data_frame = data_frame[data_frame['pKmu_PT'] > 5000]
    data_frame = data_frame[data_frame['mu1_PT'] > 2000]
    if to_plot:
        plt.hist(data_frame['pKmu_PT'], bins=50)
        plt.xlabel('pKmu_PT')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['mu1_PT'], bins=50)
        plt.xlabel('mu1_PT')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def reduce_background(data_frame):
    data_frame = clean_cuts(data_frame)
    data_frame = impact_parameter_cleaning(data_frame)
    data_frame = pid_cleaning(data_frame)
    data_frame = chi2_cleaning(data_frame)
    data_frame = data_frame[data_frame['mu1_isMuon']]
    data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT1'] < 0]
    data_frame = identify_p_k_j_psi(data_frame, False)
    data_frame = plot_pkmu_mass(data_frame, False)
    data_frame = plot_kmu_mass(data_frame, False)
    data_frame = transverse_momentum_cleaning(data_frame, False)
    data_frame = data_frame.reset_index()
    return data_frame


def b_cleaning(data_frame, to_plot=False):
    if to_plot:
        plot_columns(data_frame=data_frame, bins=200, histogram_range=[-400, 400],
                     columns_to_plot=['Kminus_TRUEID', 'proton_TRUEID', 'tauMu_TRUEID',
                                      'mu1_TRUEID'])
        plot_columns(data_frame=data_frame, bins=200, histogram_range=[-800, 800],
                     columns_to_plot=['proton_MC_MOTHER_ID', 'Kminus_MC_MOTHER_ID', 'mu1_MC_MOTHER_ID',
                                      'tauMu_MC_MOTHER_ID', 'tauMu_MC_GD_MOTHER_ID', 'Kminus_MC_GD_MOTHER_ID',
                                      'proton_MC_GD_MOTHER_ID', 'mu1_MC_GD_MOTHER_ID'])
        plot_columns(data_frame=data_frame, bins=100, histogram_range=[1, 1000],
                     columns_to_plot=['proton_MC_MOTHER_KEY', 'Kminus_MC_MOTHER_KEY', 'mu1_MC_MOTHER_KEY',
                                      'tauMu_MC_MOTHER_KEY', 'tauMu_MC_GD_MOTHER_KEY', 'Kminus_MC_GD_MOTHER_KEY'])

    # data_frame = data_frame[data_frame['proton_PIDp'] < 0.5]
    # data_frame = data_frame[data_frame['proton_PIDK'] < 0.5]
    # data_frame = data_frame[data_frame['proton_PIDp'] - data_frame['proton_PIDK'] < 1]
    # data_frame = data_frame[data_frame['proton_PIDp'] - data_frame['proton_PIDmu'] < 1]

    # data_frame = data_frame[data_frame['Lb_MC_MOTHER_ID']>500]

    # for i in range(len(data_frame)):
    #     print(data_frame['Kminus_MC_MOTHER_ID'].loc[i], data_frame['proton_MC_MOTHER_ID'].loc[i],
    #           data_frame['mu1_MC_MOTHER_ID'].loc[i], data_frame['tauMu_MC_MOTHER_ID'].loc[i])

    # data_frame = data_frame[data_frame['mu1_MC_MOTHER_ID'].abs()==511]
    # data_frame = data_frame[data_frame['tauMu_MC_MOTHER_ID'].abs()==15]
    # data_frame = data_frame[data_frame['Kminus_MC_GD_MOTHER_ID'].abs() ==511]
    # data_frame = data_frame[data_frame['proton_MC_GD_MOTHER_ID'].abs() == 511]
    # data_frame = data_frame[data_frame['tauMu_MC_GD_MOTHER_ID'].abs() == 511]
    print(len(data_frame))
    data_frame = data_frame[data_frame['Kminus_TRUEID'].abs() == 321]
    print(len(data_frame))
    data_frame = data_frame[data_frame['proton_TRUEID'].abs() == 211]
    # data_frame = data_frame[(data_frame['proton_TRUEID'].abs() == 211) | (data_frame['proton_TRUEID'].abs() == 321)]
    # data_frame = data_frame[(data_frame['Kminus_TRUEID'].abs() == 211) | (data_frame['Kminus_TRUEID'].abs() == 321)]
    print(len(data_frame))
    data_frame = data_frame[data_frame['mu1_TRUEID'].abs() == 13]
    print(len(data_frame))
    data_frame = data_frame[data_frame['tauMu_TRUEID'].abs() == 13]
    print(len(data_frame))

    print('True id tests done')
    if to_plot:
        plot_compare_data(data_frame[data_frame['proton_MC_MOTHER_ID'].abs() == 313],
                          data_frame[data_frame['proton_MC_MOTHER_ID'].abs() != 313], 20, ['proton_IPCHI2_OWNPV'],
                          signal_name='protons from Kstar')
        plt.hist(data_frame[data_frame['proton_MC_MOTHER_ID'].abs() == 313]['proton_IPCHI2_OWNPV'], bins=50,
                    range=[0, 35], label='pions from Kstar', density=True)
        plt.hist(data_frame[data_frame['proton_MC_MOTHER_ID'].abs() != 313]['proton_IPCHI2_OWNPV'], bins=50,
                    range=[0, 35], label='pions not from Kstar', alpha=0.3, density=True)
        plt.xlabel('proton_IPCHI2_OWNPV')
        plt.legend()
        plt.show()
        print(data_frame['proton_MC_GD_MOTHER_ID'].describe())
        plt.hist(data_frame[data_frame['proton_MC_MOTHER_ID'].abs() != 313]['proton_MC_MOTHER_ID'].abs(), range=[1, 600], bins=599)
        plt.xlabel('proton_MC_MOTHER_ID')
        plt.show()

    data_frame = data_frame[data_frame['proton_MC_MOTHER_ID'].abs() == 313]
    data_frame = data_frame[data_frame['Kminus_MC_MOTHER_ID'].abs() == 313]
    print(len(data_frame))
    data_frame = data_frame[(data_frame['Kminus_MC_MOTHER_ID'] == data_frame['proton_MC_MOTHER_ID']) & (
            data_frame['Kminus_MC_MOTHER_KEY'] == data_frame['proton_MC_MOTHER_KEY'])]
    print(len(data_frame))
    # data_frame = data_frame[(data_frame['Kminus_MC_MOTHER_ID'] == data_frame['proton_MC_MOTHER_ID'])]
    data_frame = data_frame[
        (data_frame['tauMu_MC_MOTHER_ID'].abs() == 15) | (data_frame['mu1_MC_MOTHER_ID'].abs() == 15)]
    print(len(data_frame))
    data_frame = data_frame[
        (data_frame['tauMu_MC_MOTHER_ID'].abs() == 511) | (data_frame['mu1_MC_MOTHER_ID'].abs() == 511)]
    print(len(data_frame))
    data_frame = data_frame[(data_frame['Kminus_MC_GD_MOTHER_ID'].abs() == 511)]
    data_frame = data_frame[(data_frame['Kminus_MC_GD_MOTHER_ID'] == data_frame['proton_MC_GD_MOTHER_ID']) & (
            data_frame['Kminus_MC_GD_MOTHER_KEY'] == data_frame['proton_MC_GD_MOTHER_KEY'])]
    data_frame = data_frame[((data_frame['Kminus_MC_GD_MOTHER_ID'] == data_frame['tauMu_MC_GD_MOTHER_ID']) & (
            data_frame['tauMu_MC_MOTHER_ID'].abs() == 15)) | (
                                    (data_frame['Kminus_MC_GD_MOTHER_ID'] == data_frame['mu1_MC_GD_MOTHER_ID']) & (
                                    data_frame['mu1_MC_MOTHER_ID'].abs() == 15))]

    data_frame = data_frame[(data_frame['tauMu_MC_MOTHER_ID'].abs() == 15)]
    print(len(data_frame))
    data_frame = data_frame[(data_frame['mu1_MC_MOTHER_ID'].abs() == 511)]
    print(len(data_frame))

    # print((data_frame['proton_MC_MOTHER_ID'].abs()).describe())
    print(len(data_frame))

    data_frame = data_frame.reset_index()
    # for i in range(len(data_frame)):
    #     if data_frame['mu1_MC_MOTHER_ID'].loc[i] == 15:
    #         for end in ['P', 'PX', 'PY', 'PZ', 'REFPX', 'REFPY', 'REFPZ']:
    #             temp = data_frame['mu1_' + end].loc[i]
    #             data_frame.loc[i, 'mu1_' + end] = data_frame['tauMu_' + end].loc[i]
    #             data_frame.loc[i, 'tauMu_' + end] = temp

    # for i in range(len(data_frame)):
    #     if data_frame['proton_TRUEID'].loc[i] == 321:
    #         for end in ['P', 'PX', 'PY', 'PZ']:
    #             temp = data_frame['proton_' + end].loc[i]
    #             data_frame.loc[i, 'proton_' + end] = data_frame['Kminus_' + end].loc[i]
    #             data_frame.loc[i, 'Kminus_' + end] = temp

    return data_frame


if __name__ == '__main__':
    a = load_data(add_branches())
    a.dropna(inplace=True)
    df = reduce_background(a)
    df = plot_kmu_mass(df, True)
    # a = b_cleaning(a, True)
    # print(zrherhehh)
    a = clean_cuts(a, False)
    # a = a[a['Lb_pmu_ISOLATION_BDT1'] < 0.2]
    a = a.reset_index()
    df = identify_p_k_j_psi(a, to_plot=False)
    # df = plot_kmu_mass(a, True)

    df = plot_pkmu_mass(a, False)
    df = identify_p_k_j_psi(a, to_plot=False)
    df = plot_kmu_mass(df, True)
    df = identify_p_k_j_psi(df, to_plot=True)
