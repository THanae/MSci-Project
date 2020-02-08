import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from data_loader import load_data, add_branches
from ip_calculations import line_point_distance
from masses import masses, get_mass
from plotting_functions import plot_compare_data, plot_columns


def analyse_pkmu_mass(data_frame, to_plot: bool):
    """
    Analyse the pkmu mass of the data set
    :param data_frame:
    :param to_plot: if True, the pkmu mass is plotted, along with other parameters
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pkmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    compare_data = data_frame[(data_frame['pkmu_mass'] < 2300) & (data_frame['pkmu_mass'] > 2200)]
    background_selection = data_frame[(data_frame['pkmu_mass'] > 2300) | (data_frame['pkmu_mass'] < 2200)]
    total_charge = np.sign(data_frame['Kminus_ID']) + np.sign(data_frame['proton_ID']) + np.sign(data_frame['mu1_ID'])
    print(total_charge.describe())
    total_charge = np.sign(data_frame['Kminus_ID']) + np.sign(data_frame['proton_ID']) + np.sign(
        data_frame['mu1_ID']) + np.sign(data_frame['tauMu_ID'])
    print(total_charge.describe(), 'total charge')
    if to_plot:
        n, b, p = plt.hist(data_frame['pkmu_mass'], bins=100, range=[1500, 5500])
        plt.vlines(masses['Lc'], 0, np.max(n) * 0.9)  # mass of lambda c
        # plt.fill_between([2190, 2310], 0, np.max(n), color='red', alpha=0.3)  # [2828, 3317]
        plt.xlabel('$m_{pK\\mu}$')
        plt.ylabel('occurrences')
        plt.show()


        selection_range = 100
        for p in [['proton', 'p'], ['Kminus', 'K'], ['mu1', 'mu'], ['tauMu', 'mu']]:
            pids = ['p', 'K', 'mu']
            pids.remove(p[1])
            plot_compare_data(compare_data, background_selection, histogram_range=selection_range,
                              columns_to_plot=[p[0] + '_PID' + p[1], [p[0] + '_PID' + p[1], p[0] + '_PID' + pids[0]],
                                               [p[0] + '_PID' + p[1], p[0] + '_PID' + pids[1]]], signal_name='lambda_0')
        potential_lc = data_frame[data_frame['pkmu_mass'] < 2300]
        artifact_region = data_frame[(data_frame['pkmu_mass'] > 2300) & (data_frame['pkmu_mass'] < 2800)]
        potential_lc = potential_lc.reset_index(drop=True)
        artifact_region = artifact_region.reset_index(drop=True)
        for _df, _name in [[potential_lc, 'potential lc region'], [artifact_region, 'artifact region']]:
            _df['vector_muTau'] = _df[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
            _df['tauMu_reference_p'] = _df[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
            _df['pkmu_endvertex_p'] = _df[['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
            _df_lpd = line_point_distance(vector=_df['vector_muTau'], vector_point=_df['tauMu_reference_p'],
                                          point=_df['pkmu_endvertex_p'])
            plt.hist(_df_lpd, range=[0, 4], bins=200)
            plt.xlabel(_name)
            plt.show()
    data_frame = data_frame[data_frame['pkmu_mass'] > 2800]
    # data_frame = data_frame[(data_frame['pkmu_mass'] < masses['Lc'] + 25) & (data_frame['pkmu_mass'] > masses['Lc'] - 25)]
    # data_frame = data_frame[(data_frame['pkmu_mass'] > 2625 -50) & (data_frame['pkmu_mass'] < 2625+50)]
    # data_frame = data_frame[(data_frame['pkmu_mass'] > 2300) & (data_frame['pkmu_mass'] < 2800)]
    return data_frame


def pp_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'proton']]
    data_frame['pp_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['pp_mass'], bins=100, range=[1800, 3000])
        plt.xlabel('$m_{pp}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def pktauMu_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['tauMu_P', 'mu']]
    data_frame['pktauMu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['pktauMu_mass'], bins=100)
        plt.xlabel('$m_{pK\\mu_{\\tau}}$')
        plt.ylabel('occurrences')
        plt.show()
    data_frame = data_frame[data_frame['pktauMu_mass'] > 2800]
    return data_frame


def kk_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'K']]
    data_frame['kk_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kk_mass'], bins=100, range=[1000, 2500])
        plt.xlabel('$m_{KK}$')
        plt.ylabel('occurrences')
        plt.show()
    data_frame = data_frame[data_frame['kk_mass'] > 1220]
    return data_frame


def kmu_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu']]
    data_frame['kmu1_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kmu1_mass'], bins=100)
        plt.xlabel('$m_{K\\mu_1}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def kpi_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi']]
    data_frame['kpi_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kpi_mass'], bins=100, range=[500, 2500])
        plt.xlabel('$m_{K\\pi}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def pmu_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'mu'], ['proton_P', 'proton']]
    data_frame['pmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['pmu_mass'], bins=100)
        plt.xlabel('$m_{pK(\\mu)}$')
        plt.ylabel('occurrences')
        plt.show()
    particles_associations = [['Kminus_P', 'mu'], ['mu1_P', 'mu']]
    data_frame['mumu1_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['mumu1_mass'], bins=100)
        plt.xlabel('$m_{K(\\mu)\\mu_1}$')
        plt.ylabel('occurrences')
        plt.show()
    particles_associations = [['Kminus_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['mumu2_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['mumu2_mass'], bins=100)
        plt.xlabel('$m_{K(\\mu)\\mu_2}$')
        plt.ylabel('occurrences')
        plt.show()

    if to_plot:
        plot_columns(data_frame, [0.25, 1], ['proton_ProbNNp', 'Kminus_ProbNNk', 'mu1_ProbNNmu', 'tauMu_ProbNNmu'], 200)
        plt.hist(data_frame['Kminus_ProbNNk'], bins=200)
        plt.xlabel('Kminus_ProbNNk')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def plot_pikmu_mass(data_frame, to_plot):
    """
    Plots the pikmu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi'], ['mu1_P', 'mu']]
    data_frame['pikmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        n, b, p = plt.hist(data_frame['pikmu_mass'], bins=100, range=[1500, 5000])
        # plt.vlines(masses['D0'], 0, np.max(n) * 0.9)  # mass of lambda c
        # plt.vlines(2010, 0, np.max(n) * 1)  # mass of lambda c
        plt.xlabel('$m_{K\\pi\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def plot_pmumu_mass(data_frame, to_plot=True):
    """
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        n, b, p = plt.hist(data_frame['pmumu_mass'], bins=70, range=[800, 5000])
        plt.xlabel('$m_{p\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def plot_pmu_mass(data_frame, to_plot):
    """
    Plots the pmu mass
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    df_to_plot = data_frame[np.sign(data_frame['proton_ID']) != np.sign(data_frame['mu1_ID'])]
    plt.hist(df_to_plot['pmu_mass'], bins=100, range=[1000, 3500])
    plt.xlabel('$m_{p\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    return data_frame


def plot_pk_mass(data_frame, to_plot=True):
    """
    Plots the pk mass
    :param data_frame:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton']]
    data_frame['pk_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        df_with_signs = data_frame[np.sign(data_frame['Kminus_ID']) != np.sign(data_frame['proton_ID'])]
        plt.hist(df_with_signs['pk_mass'], bins=120, range=[1400, 2600])  # peak 1480 to 1555
        plt.xlabel('$m_{pK}$')
        plt.ylabel('occurrences')
        plt.show()
        a = data_frame[(data_frame['pk_mass'] > 1480) & (data_frame['pk_mass'] < 1550)]
        plt.hist(a['pKmu_ENDVERTEX_CHI2'], bins=50)
        plt.xlabel('pKmu_ENDVERTEX_CHI2')
        plt.show()

    # data_frame = data_frame[(data_frame['pk_mass'] < 1480) | (data_frame['pk_mass'] > 1550)]
    # data_frame = data_frame[(data_frame['pk_mass'] < 1500) | (data_frame['pk_mass'] > 1520)]
    # data_frame = data_frame[(data_frame['pk_mass'] > 1910) | (data_frame['pk_mass'] < 1850)]
    # data_frame = data_frame[data_frame['pk_mass'] > 2100]
    return data_frame


def plot_kmu_mass(data_frame, to_plot):
    """
    Plots the kmu mass
    :param data_frame:
    :param to_plot:
    :return:
    """
    # particles_associations = [['Kminus_P', 'K'], ['tauMu_P', 'mu']]
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu']]
    data_frame['kmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)

    mu_peak = data_frame[(data_frame['dimuon_mass'] > 1620) & (data_frame['dimuon_mass'] < 1840)]
    mu_peak = mu_peak[np.sign(mu_peak['mu1_ID']) != np.sign(mu_peak['Kminus_ID'])]
    # mu_peak = mu_peak[mu_peak['tauMu_ID'] > 0]
    data_frame.loc[(np.sign(data_frame['mu1_ID']) == np.sign(data_frame['Kminus_ID'])), 'kmu_mass'] = 25000
    kmu_peak = mu_peak[(mu_peak['kmu_mass'] > 860) & (mu_peak['kmu_mass'] < 920)]
    mass_pk_peak = get_mass(data_frame=kmu_peak, particles_associations=[['Kminus_P', 'K'], ['proton_P', 'proton']])
    print(mass_pk_peak)
    print(len(mu_peak))
    print(data_frame['mu1_ID'].describe())
    print(data_frame['Kminus_ID'].describe())
    print(data_frame['proton_ID'].describe())
    if to_plot:
        plt.hist(data_frame['kmu_mass'], bins=100, range=[0, 4000])
        plt.xlabel('$m_{K\\mu}$')
        plt.ylabel('occurrences')
        plt.show()

        plt.hist(mu_peak['kmu_mass'], bins=60, range=[800, 3200])
        plt.xlabel('$m_{K\\mu}$')
        plt.ylabel('occurrences')
        plt.title('Kmu mass for the dimuon peak')
        plt.show()

        plt.hist(mass_pk_peak, bins=100)
        plt.xlabel('$m_{Kp}$')
        plt.ylabel('occurrences')
        plt.title('Kp mass for the kmu peak')
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
    """
    Plots the dimuon mass
    :param data_frame:
    :return:
    """
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
    """
    Applies jpsi veto
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['dimuon_mass'] = sum_m
    compare_data = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3050)]
    print('jpsi events', len(compare_data))
    compare_data = compare_data[(compare_data['Lb_M'] < 5650) & (compare_data['Lb_M'] > 5590)]
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3150) | (data_frame['dimuon_mass'] < 3050)]
    data_frame = data_frame[(data_frame['dimuon_mass'] > masses['J/psi'] + 50) | (data_frame['dimuon_mass'] < masses['J/psi'] - 250)]
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3200) | (data_frame['dimuon_mass'] < 3000)]
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3750) | (data_frame['dimuon_mass'] < 3600)]
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3700) | (data_frame['dimuon_mass'] < 3650)]
    data_frame = data_frame[(data_frame['dimuon_mass'] > masses['psi(2S)'] + 50) | (data_frame['dimuon_mass'] < masses['psi(2S)'] - 50)]
    # background_selection = data_frame[(data_frame['Lb_M'] > 5800) | (data_frame['Lb_M'] < 5200)]
    background_selection = data_frame[(data_frame['Lb_M'] > 5800)]
    print('comp', len(compare_data), 'back', len(background_selection))
    if to_plot:
        plt.hist(compare_data['pKmu_ENDVERTEX_CHI2'], bins=100, density=True, label='J/Psi data', range=[0, 50])
        plt.hist(background_selection['pKmu_ENDVERTEX_CHI2'], bins=100, density=True, alpha=0.3, label='background',
                 range=[0, 50])
        plt.xlabel('pKmu_ENDVERTEX_CHI2')
        plt.legend()
        plt.show()
        plt.hist(compare_data['Lb_ENDVERTEX_CHI2'], bins=100, density=True, label='J/Psi data', range=[0, 10])
        plt.hist(background_selection['Lb_ENDVERTEX_CHI2'], bins=100, density=True, alpha=0.3, label='background', range=[0, 10])
        plt.xlabel('Lb_ENDVERTEX_CHI2')
        plt.legend()
        plt.show()

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

        plt.hist(compare_data['Lb_FD_OWNPV'], bins=100, range=[0, 100])
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

        plt.hist2d(compare_data['mu1_PIDmu'], compare_data['tauMu_PIDmu'], bins=50, range=[[0, 15], [0, 15]],
                   norm=LogNorm())
        plt.xlabel('mu1_PIDmu')
        plt.ylabel('tauMu_PIDmu')
        plt.colorbar()
        plt.show()
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

        fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 0.5})
        axs[0].hist(compare_data['pKmu_ENDVERTEX_CHI2'], bins=50, range=[0, 100])
        axs[0].set_xlabel('pKmu_ENDVERTEX_CHI2')
        axs[0].set_ylabel('occurrences')
        axs[0].set_title('JPSI')
        axs[1].hist(background_selection['pKmu_ENDVERTEX_CHI2'], bins=50, range=[0, 100])
        axs[1].set_xlabel('pKmu_ENDVERTEX_CHI2')
        axs[1].set_ylabel('occurrences')
        axs[1].set_title('Background')
        plt.show()
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
    # data_frame = data_frame[data_frame['Kminus_PIDK'] > 20]  # or 9?
    # data_frame = data_frame[data_frame['proton_PIDp'] > 40]  # or 9?
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
    print('cuts cleaned', len(data_frame))
    data_frame = identify_p_k_j_psi(data_frame, False)
    print('j/psi cleaning', len(data_frame))
    data_frame = impact_parameter_cleaning(data_frame)
    print('impact parameter cleaning', len(data_frame))
    data_frame = pid_cleaning(data_frame)
    print('PID cleaning', len(data_frame))
    data_frame = chi2_cleaning(data_frame)
    print('chi squared cleaning', len(data_frame))
    data_frame = data_frame[data_frame['mu1_isMuon']]
    print('mu1_isMuon cleaning', len(data_frame))
    data_frame = data_frame[data_frame['tauMu_isMuon']]
    print('tauMu_isMuon cleaning', len(data_frame))
    data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT1'] < -0]
    print('isolation angle cleaning', len(data_frame))
    data_frame = transverse_momentum_cleaning(data_frame, False)
    print('transverse momentum cleaning', len(data_frame))
    # data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT2'] < -0.5]
    # data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT3'] < -0.5]
    # data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT4'] < -0.5]
    # data_frame = data_frame[data_frame['Lb_pmu_TR1_PIDmu'] <= 0]
    # data_frame = data_frame[data_frame['Lb_pmu_TR1_PIDK'] <= -8]
    # data_frame = data_frame[data_frame['Lb_pmu_TR1_PIDp'] <= -7]
    # data_frame = data_frame[(data_frame['Lb_pmu_TR1_PIDK'] > -2) & (data_frame['Lb_pmu_TR1_PIDK'] <2)]
    # data_frame = data_frame[(data_frame['Lb_pmu_TR1_PIDp'] > -3) & (data_frame['Lb_pmu_TR1_PIDp'] <1)]
    # plt.hist(data_frame['Lb_pmu_ISOLATION_BDT2'])
    # plt.show()
    data_frame = identify_p_k_j_psi(data_frame, False)
    print('j/psi cleaning', len(data_frame))
    data_frame = analyse_pkmu_mass(data_frame, False)
    print('Lc cleaning', len(data_frame))
    data_frame = pktauMu_mass(data_frame, False)
    # data_frame = data_frame[data_frame['Lb_ENDVERTEX_CHI2']<1]
    # plt.hist(data_frame['Lb_ENDVERTEX_CHI2'], bins=50)
    # plt.show()
    # data_frame = kk_mass(data_frame, False)
    # data_frame = plot_kmu_mass(data_frame, False)
    # data_frame = plot_pk_mass(data_frame, False)
    # data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT1'] ==-2]
    # print(len(data_frame))
    # data_frame = data_frame.reset_index(drop=True)
    # data_frame['vector_muTau'] = data_frame[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    # data_frame['tauMu_reference_point'] = data_frame[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    # data_frame['pkmu_endvertex_point'] = data_frame[['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    # data_frame['impact_parameter_thingy'] = line_point_distance(vector=data_frame['vector_muTau'],
    #                                                             vector_point=data_frame['tauMu_reference_point'],
    #                                                             point=data_frame['pkmu_endvertex_point'])
    # data_frame = data_frame[data_frame['impact_parameter_thingy'] > 0.05]
    data_frame = data_frame.reset_index(drop=True)
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
        plot_columns(data_frame=data_frame, bins=100, histogram_range=[-10, 10],
                     columns_to_plot=['Kminus_TRACK_Type', 'proton_TRACK_Type', 'mu1_TRACK_Type', 'tauMu_TRACK_Type'])
        plot_columns(data_frame=data_frame, bins=100, histogram_range=[0, 100],
                     columns_to_plot=['Kminus_TRACK_Key', 'proton_TRACK_Key', 'mu1_TRACK_Key', 'tauMu_TRACK_Key'])

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
        plt.hist(data_frame[data_frame['proton_MC_MOTHER_ID'].abs() != 313]['proton_MC_MOTHER_ID'].abs(),
                 range=[1, 600], bins=599)
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
    # analyse_pkmu_mass(a, True)
    # a = a[a['Lb_pmu_ISOLATION_BDT1'] > 0.7]
    # plot_columns(a, [-75, 75], ['Lb_pmu_TR1_PIDp', 'Lb_pmu_TR1_PIDK', 'Lb_pmu_TR1_PIDmu', 'Lb_pmu_TR1_PIDpi'], 100)
    df = reduce_background(a)
    pp_mass(data_frame=df, to_plot=True)
    # pktauMu_mass(df, True)
    kpi_mass(df, True)
    kmu_mass(df, True)
    plot_pikmu_mass(df, True)
    kk_mass(df, True)
    pmu_mass(df, True)
    df = plot_pk_mass(df)

    # df = b_cleaning(a)
    # plot_columns(df, [-50, 50], ['Lb_pmu_TR1_PIDp', 'Lb_pmu_TR1_PIDK', 'Lb_pmu_TR1_PIDmu', 'Lb_pmu_TR1_PIDpi'], 100)
    # plt.hist2d(df['Kminus_PIDK'], df['proton_PIDp'], range=[[15, 70], [15, 70]], bins=40)
    # plt.xlabel('Kminus_PIDK')
    # plt.ylabel('proton_PIDp')
    # plt.show()
    # df = df[(df['dimuon_mass'] > 1620) & (df['dimuon_mass'] < 1840)]  # dimuon peak
    # print(len(df))
    analyse_pkmu_mass(df, True)
    # df = plot_kmu_mass(df, True)
    # identify_p_k_j_psi(df, to_plot=True)
    # df = plot_pmumu_mass(df, True)
    df = plot_pmu_mass(df, True)

    # plot_pikmu_mass(df, True)
    # a = b_cleaning(a, True)
    a = clean_cuts(a, False)
    # a = a[a['Lb_pmu_ISOLATION_BDT1'] < 0.2]
    a = a.reset_index()
    df = identify_p_k_j_psi(a, to_plot=False)
    # df = plot_kmu_mass(a, True)

    df = analyse_pkmu_mass(a, False)
    df = identify_p_k_j_psi(a, to_plot=True)
    df = plot_kmu_mass(df, True)
    df = identify_p_k_j_psi(df, to_plot=True)
