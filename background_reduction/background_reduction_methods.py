import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ip_calculations import line_point_distance
from masses import masses, get_mass
from plotting_functions import plot_compare_data, plot_columns


def ip_star_cleaning(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Cuts on IP*, which is the impact parameter between the muon from the supposed tau and the pkmu vertex
    :param data_frame: data frame to cut on
    :return:
    """
    data_frame['vector_muTau'] = data_frame[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    data_frame['tauMu_reference_point'] = data_frame[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    data_frame['pkmu_endvertex_point'] = data_frame[
        ['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    data_frame['pkmu_direction'] = data_frame[['pKmu_PX', 'pKmu_PY', 'pKmu_PZ']].values.tolist()
    data_frame['impact_parameter_thingy'] = line_point_distance(vector=data_frame['vector_muTau'],
                                                                vector_point=data_frame['tauMu_reference_point'],
                                                                point=data_frame['pkmu_endvertex_point'],
                                                                direction=data_frame['pkmu_direction'])
    data_frame = data_frame[data_frame['impact_parameter_thingy'] > -0.02]
    return data_frame


def kmu_cut(data_frame: pd.DataFrame, to_plot: bool = False) -> pd.DataFrame:
    """
    Cuts on the kmu mass, where we assume a decay of the type Lb -> p D0 (-> K mu nu) mu nu
    If to_plot is true, we also show the Kpi mass plot (where we assume Lb -> p D0 (-> K pi) pi)
    :param data_frame: data frame to cut on
    :param to_plot: if True, plots the Kpi mass and the Kmu mass, where the particles have opposite charges
    :return:
    """
    data_frame['kpi1_mass'] = get_mass(data_frame, particles_associations=[['Kminus_P', 'K'], ['mu1_P', 'pi']])
    data_frame['ktaupi_mass'] = get_mass(data_frame, particles_associations=[['Kminus_P', 'K'], ['tauMu_P', 'pi']])
    data_frame['kmu1_mass'] = get_mass(data_frame, particles_associations=[['Kminus_P', 'K'], ['mu1_P', 'mu']])
    data_frame['ktauMu_mass'] = get_mass(data_frame, particles_associations=[['Kminus_P', 'K'], ['tauMu_P', 'mu']])
    if to_plot:
        plt.hist([data_frame[np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['mu1_ID'])]['kpi1_mass'],
                  data_frame[np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['tauMu_ID'])]['ktaupi_mass']],
                 bins=100, stacked=True, color=['C0', 'C0'])
        plt.axvline(masses['D0'], c='k')
        plt.xlabel('$m_{K\\pi}$')
        plt.show()
        plt.hist([data_frame[np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['mu1_ID'])]['kmu1_mass'],
                  data_frame[np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['tauMu_ID'])]['ktauMu_mass']],
                 bins=100, stacked=True, color=['C0', 'C0'])
        plt.axvline(masses['D0'], c='k')
        plt.xlabel('$m_{K\\mu}$')
        plt.show()
    to_drop_1 = data_frame[
        (np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['mu1_ID'])) & (data_frame['kmu1_mass'] < masses['D0'])]
    to_drop_2 = data_frame[(np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['tauMu_ID'])) & (
            data_frame['ktauMu_mass'] < masses['D0'])]
    data_frame = data_frame.drop(list(to_drop_1.index))
    data_frame = data_frame.drop(list(to_drop_2.index))
    return data_frame


def remove_high_pkmu_mass(data_frame):
    """
    Removes pkmu masses high enough so that no tau can be present in the decay
    This cut is only applied on the mass of p, K and mu1, as we do nto expect mu1 to come from a tau
    :param data_frame: data frame to cut on
    :return:
    """
    data_frame['pkmu'] = get_mass(data_frame,
                                  particles_associations=[['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']])
    to_drop_1 = data_frame[
        (np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['mu1_ID'])) & (
                data_frame['pkmu'] > masses['Lb'] - masses['tau'])]
    data_frame = data_frame.drop(list(to_drop_1.index))

    # to_drop_2 = data_frame[
    #     (np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['tauMu_ID'])) & (data_frame['pktauMu'] > 3839)]
    # data_frame = data_frame.drop(list(to_drop_2.index))
    return data_frame


def analyse_pkmu_for_2_muons(data_frame, pkmu_threshold: int = 2800, to_plot: bool = False):
    """
    Analyse the pkmu mass of the data set and cut on the pkmu mass
    :param data_frame: data frame to cut on
    :param pkmu_threshold: threshold at which to cut the pkmu mass
    :param to_plot: if True, the pkmu mass is plotted, along with other parameters
    :return:
    """
    data_frame = data_frame.reset_index(drop=True)
    data_frame['vector_muTau'] = data_frame[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    data_frame['tauMu_reference_point'] = data_frame[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    data_frame['pkmu_endvertex_point'] = data_frame[
        ['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    data_frame['pkmu_direction'] = data_frame[['pKmu_PX', 'pKmu_PY', 'pKmu_PZ']].values.tolist()
    data_frame['ip_tauMu'] = line_point_distance(vector=data_frame['vector_muTau'],
                                                 vector_point=data_frame['tauMu_reference_point'],
                                                 point=data_frame['pkmu_endvertex_point'],
                                                 direction=data_frame['pkmu_direction'])

    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton']]
    data_frame['pk_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pkmu_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['tauMu_P', 'mu']]
    data_frame['pkmu_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu']]
    data_frame['kmu_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['tauMu_P', 'mu']]
    data_frame['kmu_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    pkmu_plus = data_frame[(data_frame['proton_ID'] == 2212) & (data_frame['Kminus_ID'] == -321)]
    pkmu_plus.loc[:, 'pkmu_mass'] = pkmu_plus.loc[:, 'pkmu_mass1']
    pkmu_plus.loc[:, 'kmu_mass'] = pkmu_plus.loc[:, 'kmu_mass1']
    index_mask = (pkmu_plus[pkmu_plus['tauMu_ID'] == -13]).index
    pkmu_plus.loc[index_mask, 'pkmu_mass'] = pkmu_plus.loc[index_mask, 'pkmu_mass2']
    pkmu_plus.loc[index_mask, 'kmu_mass'] = pkmu_plus.loc[index_mask, 'kmu_mass2']
    pkmu_plus.loc[index_mask, 'mu_PID'] = pkmu_plus.loc[index_mask, 'tauMu_PIDmu']
    pkmu_minus = data_frame[(data_frame['proton_ID'] == -2212) & (data_frame['Kminus_ID'] == +321)]
    pkmu_minus.loc[:, 'pkmu_mass'] = pkmu_minus.loc[:, 'pkmu_mass1']
    pkmu_minus.loc[:, 'kmu_mass'] = pkmu_minus.loc[:, 'kmu_mass1']
    index_mask = (pkmu_minus[pkmu_minus['tauMu_ID'] == +13]).index
    pkmu_minus.loc[index_mask, 'pkmu_mass'] = pkmu_minus.loc[index_mask, 'pkmu_mass2']
    pkmu_minus.loc[index_mask, 'kmu_mass'] = pkmu_minus.loc[index_mask, 'kmu_mass2']

    if to_plot:
        _range, _bins = [-0.4, 0.4], 100
        plt.hist(np.load(f'C:\\Users\\Hanae\\Documents\\MSci Project\\MsciCode\\ipstar_jpsi_sign.npy'),
                 label='jpsi data',
                 density=True, alpha=0.3, bins=_bins)
        plt.hist(pkmu_plus[(pkmu_plus['pkmu_mass'] < 2400)]['ip_tauMu'], label='less than 2.4GeV',
                 density=True, histtype='step', bins=_bins)
        plt.hist(pkmu_plus[(pkmu_plus['pkmu_mass'] > 2400) & (pkmu_plus['pkmu_mass'] < 2800)][
                     'ip_tauMu'], label='2.4GeV to 2.8GeV', histtype='step', density=True, bins=_bins)
        plt.hist(pkmu_plus[pkmu_plus['pkmu_mass'] > 2800]['ip_tauMu'], label='more than 2.8GeV',
                 histtype='step', density=True, bins=_bins)
        plt.title('IP* normalised distribution for different parts of the data')
        plt.xlim(_range)
        plt.legend()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist2d(pkmu_plus['Lb_M'], pkmu_plus['pkmu_mass'], bins=100, range=[[2200, 6000], [1640, 3500]])
        ax1.set_xlabel('Lb_M')
        ax1.set_ylabel('$m_{pK\\mu}$')
        ax2.hist2d(pkmu_minus['Lb_M'], pkmu_minus['pkmu_mass'], bins=100, range=[[2200, 6000], [1640, 3500]])
        ax2.set_xlabel('Lb_M')
        ax2.set_ylabel('$m_{pK\\mu}$')
        plt.show()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(pkmu_plus['pkmu_mass'], bins=100, range=[1500, 5000])
        ax1.set_xlabel('$m_{p^{+}K^{-}\\mu^{+}}$')
        ax2.hist(pkmu_minus['pkmu_mass'], bins=100, range=[1500, 5000])
        ax2.set_xlabel('$m_{p^{-}K^{+}\\mu^{-}}$')
        plt.show()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(pkmu_plus['pk_mass'], bins=100, range=[1400, 2400])
        ax1.set_xlabel('$m_{p^{+}K^{-}}$')
        ax2.hist(pkmu_minus['pk_mass'], bins=100, range=[1400, 2400])
        ax2.set_xlabel('$m_{p^{-}K^{+}}$')
        plt.show()

        mask_pk_plus = (pkmu_plus['pk_mass'] > 1505) & (pkmu_plus['pk_mass'] < 1522.5)
        mask_pk_minus = (pkmu_minus['pk_mass'] > 1505) & (pkmu_minus['pk_mass'] < 1522.5)
        mask_pkmu_plus = (pkmu_plus['pkmu_mass'] > 2275) & (pkmu_plus['pkmu_mass'] < 2297.5)
        mask_pkmu_minus = (pkmu_minus['pkmu_mass'] > 2275) & (pkmu_minus['pkmu_mass'] < 2297.5)
        mask_combined_plus = mask_pk_plus & mask_pkmu_plus
        mask_combined_minus = mask_pk_minus & mask_pkmu_minus
        print(len(data_frame))
        print(len(pkmu_plus[mask_pk_plus]), len(pkmu_plus[mask_pkmu_plus]), len(pkmu_plus[mask_combined_plus]))
        print(len(pkmu_minus[mask_pk_minus]), len(pkmu_minus[mask_pkmu_minus]), len(pkmu_minus[mask_combined_minus]))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist2d(pkmu_plus['pk_mass'], pkmu_plus['pkmu_mass'], bins=40, range=[[1400, 2100], [1600, 2500]])
        ax1.set_ylabel('$m_{p^{+}K^{-}\\mu^{+}}$')
        ax1.set_xlabel('$m_{p^{+}K^{-}}$')
        ax2.hist2d(pkmu_minus['pk_mass'], pkmu_minus['pkmu_mass'], bins=40, range=[[1400, 2100], [1600, 2500]])
        ax2.set_ylabel('$m_{p^{-}K^{+}\\mu^{-}}$')
        ax2.set_xlabel('$m_{p^{-}K^{+}}$')
        plt.show()

    print((pkmu_minus['pkmu_mass']).describe())
    data_frame = data_frame.drop(list((pkmu_minus[pkmu_minus['pkmu_mass'] < pkmu_threshold]).index))
    data_frame = data_frame.drop(list((pkmu_plus[pkmu_plus['pkmu_mass'] < pkmu_threshold]).index))
    return data_frame


def identify_p_k_j_psi(data_frame, to_plot=True):
    """
    Applies jpsi veto
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['dimuon_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    compare_data = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3050)]
    print('jpsi events', len(compare_data))
    compare_data = compare_data[(compare_data['Lb_M'] < 5650) & (compare_data['Lb_M'] > 5590)]
    print('jpsi events from Lb', len(compare_data))
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3150) | (data_frame['dimuon_mass'] < 3050)]
    # data_frame = data_frame[
    #     (data_frame['dimuon_mass'] > masses['J/psi'] + 50) | (data_frame['dimuon_mass'] < masses['J/psi'] - 250)]
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3200) | (data_frame['dimuon_mass'] < 3000)]
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3750) | (data_frame['dimuon_mass'] < 3600)]
    # data_frame = data_frame[(data_frame['dimuon_mass'] > 3700) | (data_frame['dimuon_mass'] < 3650)]
    # data_frame = data_frame[
    #     (data_frame['dimuon_mass'] > masses['psi(2S)'] + 50) | (data_frame['dimuon_mass'] < masses['psi(2S)'] - 50)]
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
        plt.hist(background_selection['Lb_ENDVERTEX_CHI2'], bins=100, density=True, alpha=0.3, label='background',
                 range=[0, 10])
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

        n, b, t = plt.hist(data_frame['dimuon_mass'], bins=100, range=[0, 5000])
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

    data_frame = data_frame[
        (data_frame['dimuon_mass'] > masses['J/psi'] + 50) | (data_frame['dimuon_mass'] < masses['J/psi'] - 250)]
    data_frame = data_frame[
        (data_frame['dimuon_mass'] > masses['psi(2S)'] + 50) | (data_frame['dimuon_mass'] < masses['psi(2S)'] - 50)]
    return data_frame
    # return compare_data


def clean_cuts(data_frame, to_plot=False):
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

    data_frame = data_frame[(data_frame['tauMu_PIDmu'] > 5)]
    data_frame = data_frame[(data_frame['mu1_PIDmu'] > 5)]

    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def chi2_cleaning(data_frame):
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
    data_frame = data_frame[data_frame['Kminus_PIDK'] - data_frame['Kminus_PIDp'] > 10]  # or 9?
    data_frame = data_frame[data_frame['Kminus_PIDK'] - data_frame['Kminus_PIDmu'] > 10]  # or 9?
    data_frame = data_frame[data_frame['Kminus_PIDK'] > 10]  # or 9?
    data_frame = data_frame[data_frame['proton_PIDp'] > 25]  # or 9?
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
