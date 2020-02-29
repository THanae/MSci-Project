import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ip_calculations import line_point_distance
from masses import masses, get_mass
from plotting_functions import plot_compare_data, plot_columns


def ip_star_cleaning(data_frame):
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


def analyse_pkmu_for_2_muons(data_frame, to_plot: bool):
    """
    Analyse the pkmu mass of the data set
    :param data_frame:
    :param to_plot: if True, the pkmu mass is plotted, along with other parameters
    :return:
    """
    print(len(data_frame))
    print(len(data_frame[np.sign(data_frame['proton_ID']) != np.sign(data_frame['Kminus_ID'])]))
    print(len(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['tauMu_ID'])]))
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
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'K']]
    data_frame['pkswap_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pkmu_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['tauMu_P', 'mu']]
    data_frame['pkmu_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'K'], ['mu1_P', 'mu']]
    data_frame['pkmuswap_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'K'], ['tauMu_P', 'mu']]
    data_frame['pkmuswap_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pmu_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['proton_P', 'proton'], ['tauMu_P', 'mu']]
    data_frame['pmu_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu']]
    data_frame['kmu_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['tauMu_P', 'mu']]
    data_frame['kmu_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    pkmu_mass_lc_plus = data_frame[(data_frame['proton_ID'] == 2212) & (data_frame['Kminus_ID'] == -321)]
    pkmu_mass_lc_plus.loc[:, 'pkmu_mass'] = pkmu_mass_lc_plus.loc[:, 'pkmu_mass1']
    pkmu_mass_lc_plus.loc[:, 'pkmuswap_mass'] = pkmu_mass_lc_plus.loc[:, 'pkmuswap_mass1']
    pkmu_mass_lc_plus.loc[:, 'pmu_mass'] = pkmu_mass_lc_plus.loc[:, 'pmu_mass1']
    pkmu_mass_lc_plus.loc[:, 'kmu_mass'] = pkmu_mass_lc_plus.loc[:, 'kmu_mass1']
    pkmu_mass_lc_plus.loc[:, 'mu_PID'] = pkmu_mass_lc_plus.loc[:, 'mu1_PIDmu']
    index_mask = (pkmu_mass_lc_plus[pkmu_mass_lc_plus['tauMu_ID'] == -13]).index
    pkmu_mass_lc_plus.loc[index_mask, 'pkmu_mass'] = pkmu_mass_lc_plus.loc[index_mask, 'pkmu_mass2']
    pkmu_mass_lc_plus.loc[index_mask, 'pkmuswap_mass'] = pkmu_mass_lc_plus.loc[index_mask, 'pkmuswap_mass2']
    pkmu_mass_lc_plus.loc[index_mask, 'pmu_mass'] = pkmu_mass_lc_plus.loc[index_mask, 'pmu_mass2']
    pkmu_mass_lc_plus.loc[index_mask, 'kmu_mass'] = pkmu_mass_lc_plus.loc[index_mask, 'kmu_mass2']
    pkmu_mass_lc_plus.loc[index_mask, 'mu_PID'] = pkmu_mass_lc_plus.loc[index_mask, 'tauMu_PIDmu']
    pkmu_mass_lc_minus = data_frame[(data_frame['proton_ID'] == -2212) & (data_frame['Kminus_ID'] == +321)]
    pkmu_mass_lc_minus.loc[:, 'pkmu_mass'] = pkmu_mass_lc_minus.loc[:, 'pkmu_mass1']
    pkmu_mass_lc_minus.loc[:, 'pkmuswap_mass'] = pkmu_mass_lc_minus.loc[:, 'pkmuswap_mass1']
    pkmu_mass_lc_minus.loc[:, 'pmu_mass'] = pkmu_mass_lc_minus.loc[:, 'pmu_mass1']
    pkmu_mass_lc_minus.loc[:, 'kmu_mass'] = pkmu_mass_lc_minus.loc[:, 'kmu_mass1']
    index_mask = (pkmu_mass_lc_minus[pkmu_mass_lc_minus['tauMu_ID'] == +13]).index
    pkmu_mass_lc_minus.loc[index_mask, 'pkmu_mass'] = pkmu_mass_lc_minus.loc[index_mask, 'pkmu_mass2']
    pkmu_mass_lc_minus.loc[index_mask, 'pkmuswap_mass'] = pkmu_mass_lc_minus.loc[index_mask, 'pkmuswap_mass2']
    pkmu_mass_lc_minus.loc[index_mask, 'pmu_mass'] = pkmu_mass_lc_minus.loc[index_mask, 'pmu_mass2']
    pkmu_mass_lc_minus.loc[index_mask, 'kmu_mass'] = pkmu_mass_lc_minus.loc[index_mask, 'kmu_mass2']

    _range = [-0.4, 0.4]
    _bins=100
    plt.hist(np.load(f'C:\\Users\\Hanae\\Documents\\MSci Project\\MsciCode\\ipstar_jpsi_sign.npy'), label='jpsi data',
             density=True, alpha=0.3, bins=_bins)
    plt.hist(pkmu_mass_lc_plus[(pkmu_mass_lc_plus['pkmu_mass'] < 2400)]['ip_tauMu'], label='less than 2.4GeV',
             density=True, histtype='step', bins=_bins)
    plt.hist(pkmu_mass_lc_plus[(pkmu_mass_lc_plus['pkmu_mass'] > 2400) & (pkmu_mass_lc_plus['pkmu_mass'] < 2800)][
                 'ip_tauMu'], label='2.4GeV to 2.8GeV', histtype='step', density=True, bins=_bins)
    plt.hist(pkmu_mass_lc_plus[pkmu_mass_lc_plus['pkmu_mass'] > 2800]['ip_tauMu'], label='more than 2.8GeV',
             histtype='step', density=True, bins=_bins)
    plt.title('IP* normalised distribution for different parts of the data')
    plt.xlim(_range)
    plt.legend()
    plt.show()

    # # pkmu_mass_lc_plus = pkmu_mass_lc_plus[pkmu_mass_lc_plus['pkmu_mass'] < 2800]
    # plt.hist(pkmu_mass_lc_plus['mu_PID'], bins=50)
    # plt.xlabel('mu- PIDmu')
    # plt.show()
    # plt.hist2d(pkmu_mass_lc_plus['Lb_M'], pkmu_mass_lc_plus['pkmu_mass'], bins=100, range=[[2200, 6000], [1640, 3500]])
    # plt.xlabel('Lb_M')
    # plt.ylabel('$m_{pK\\mu}$')
    # plt.show()
    # plt.hist2d(pkmu_mass_lc_minus['Lb_M'], pkmu_mass_lc_minus['pkmu_mass'], bins=100, range=[[2200, 6000], [1640, 3500]])
    # plt.xlabel('Lb_M')
    # plt.ylabel('$m_{pK\\mu}$')
    # plt.show()
    # plt.hist(data_frame['pKmu_M'], bins=100, range=[1500, 5000])
    # plt.xlabel('pKmu_M')
    # plt.show()
    # plt.hist(pkmu_mass_lc_plus['pkmu_mass'], bins=100, range=[1500, 5000])
    # plt.xlabel('$m_{p^{+}K^{-}\\mu^{-}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_minus['pkmu_mass'], bins=100, range=[1500, 5000])
    # plt.xlabel('$m_{p^{-}K^{+}\\mu^{+}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_plus['pkmuswap_mass'], bins=100, range=[1500, 5000])
    # plt.xlabel('$m_{p^{+}K^{-}\\mu^{-}}$ swap')
    # plt.show()
    # plt.hist(pkmu_mass_lc_minus['pkmuswap_mass'], bins=100, range=[1500, 5000])
    # plt.xlabel('$m_{p^{-}K^{+}\\mu^{+}}$ swap')
    # plt.show()
    # plt.hist(pkmu_mass_lc_plus['pmu_mass'], bins=100, range=[1000, 2500])
    # plt.xlabel('$m_{p^{+}\\mu^{-}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_minus['pmu_mass'], bins=100, range=[1000, 2500])
    # plt.xlabel('$m_{p^{-}\\mu^{+}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_plus['pk_mass'], bins=100, range=[1400, 2400])
    # plt.xlabel('$m_{p^{+}K^{-}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_minus['pk_mass'], bins=100, range=[1400, 2400])
    # plt.xlabel('$m_{p^{-}K^{+}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_plus['kmu_mass'], bins=100, range=[400, 2400])
    # plt.xlabel('$m_{k^{-}\\mu^{-}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_minus['kmu_mass'], bins=100, range=[400, 2400])
    # plt.xlabel('$m_{k^{+}\\mu^{+}}$')
    # plt.show()
    # print(len(data_frame))
    # print(len(pkmu_mass_lc_plus[(pkmu_mass_lc_plus['pk_mass'] > 1505) & (pkmu_mass_lc_plus['pk_mass'] < 1522.5)]))
    # print(len(pkmu_mass_lc_plus[(pkmu_mass_lc_plus['pkmu_mass'] > 2275) & (pkmu_mass_lc_plus['pkmu_mass'] < 2297.5)]))
    # print(len(pkmu_mass_lc_plus[
    #               ((pkmu_mass_lc_plus['pkmu_mass'] > 2275) & (pkmu_mass_lc_plus['pkmu_mass'] < 2297.5)) & (
    #                           (pkmu_mass_lc_plus['pk_mass'] > 1505) & (pkmu_mass_lc_plus['pk_mass'] < 1522.5))]))
    # print(len(pkmu_mass_lc_minus[(pkmu_mass_lc_minus['pk_mass'] > 1505) & (pkmu_mass_lc_minus['pk_mass'] < 1522.5)]))
    # print(len(pkmu_mass_lc_minus[(pkmu_mass_lc_minus['pkmu_mass'] > 2275) & (pkmu_mass_lc_minus['pkmu_mass'] < 2297.5)]))
    # print(len(pkmu_mass_lc_minus[
    #               ((pkmu_mass_lc_minus['pkmu_mass'] > 2275) & (pkmu_mass_lc_minus['pkmu_mass'] < 2297.5)) & (
    #                       (pkmu_mass_lc_minus['pk_mass'] > 1505) & (pkmu_mass_lc_minus['pk_mass'] < 1522.5))]))
    # plt.hist2d(pkmu_mass_lc_plus['pk_mass'], pkmu_mass_lc_plus['pkmu_mass'], bins=40,
    #            range=[[1400, 2100], [1600, 2500]])
    # plt.ylabel('$m_{p^{+}K^{-}\\mu^{+}}$')
    # plt.xlabel('$m_{p^{+}K^{-}}$')
    # plt.show()
    # plt.hist2d(pkmu_mass_lc_minus['pk_mass'], pkmu_mass_lc_minus['pkmu_mass'], bins=40,
    #            range=[[1400, 2100], [1600, 2500]])
    # plt.ylabel('$m_{p^{-}K^{+}\\mu^{-}}$')
    # plt.xlabel('$m_{p^{-}K^{+}}$')
    # plt.show()
    # plt.hist2d(pkmu_mass_lc_plus['kmu_mass'], pkmu_mass_lc_plus['pkmu_mass'], bins=50,
    #            range=[[400, 2400], [1500, 2500]])
    # plt.ylabel('$m_{K^{-}\\mu^{-}}$')
    # plt.xlabel('$m_{p^{+}K^{-}\\mu^{-}}$')
    # plt.show()
    # plt.hist2d(pkmu_mass_lc_minus['kmu_mass'], pkmu_mass_lc_minus['pkmu_mass'], bins=50,
    #            range=[[400, 2400], [1500, 2500]])
    # plt.ylabel('$m_{K^{+}\\mu^{+}}$')
    # plt.xlabel('$m_{p^{-}K^{+}\\mu^{+}}$')
    # plt.show()

    # particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    # data_frame['pkmu_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    # particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['tauMu_P', 'mu']]
    # data_frame['pkmu_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    # pkmu_mass_lc_plus1 = data_frame[(data_frame['proton_ID'] == 2212) & (data_frame['Kminus_ID'] == -321)]
    # pkmu_mass_lc_plus1 = pkmu_mass_lc_plus1[(pkmu_mass_lc_plus1['mu1_ID'] == -13)]
    # pkmu_mass_lc_plus1['pkmu_mass'] = pkmu_mass_lc_plus1['pkmu_mass1']
    # pkmu_mass_lc_minus1 = data_frame[(data_frame['proton_ID'] == -2212) & (data_frame['Kminus_ID'] == 321)]
    # pkmu_mass_lc_minus1 = pkmu_mass_lc_minus1[(pkmu_mass_lc_minus1['mu1_ID'] == +13)]
    # pkmu_mass_lc_minus1['pkmu_mass'] = pkmu_mass_lc_minus1['pkmu_mass1']
    # pkmu_mass_lc_plus2 = data_frame[(data_frame['proton_ID'] == 2212) & (data_frame['Kminus_ID'] == -321)]
    # pkmu_mass_lc_plus2 = pkmu_mass_lc_plus2[(pkmu_mass_lc_plus2['tauMu_ID'] == -13)]
    # pkmu_mass_lc_plus2['pkmu_mass'] = pkmu_mass_lc_plus2['pkmu_mass2']
    # pkmu_mass_lc_minus2 = data_frame[(data_frame['proton_ID'] == -2212) & (data_frame['Kminus_ID'] == 321)]
    # pkmu_mass_lc_minus2 = pkmu_mass_lc_minus2[(pkmu_mass_lc_minus2['tauMu_ID'] == +13)]
    # pkmu_mass_lc_minus2['pkmu_mass'] = pkmu_mass_lc_minus2['pkmu_mass2']
    # plt.hist(pkmu_mass_lc_minus1['pkmu_mass'], bins=100, range=[1500, 5000])
    # plt.xlabel('$m_{p^{+}K^{-}\\mu^{-}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_plus1['pkmu_mass'], bins=100, range=[1500, 5000])
    # plt.xlabel('$m_{p^{-}K^{+}\\mu^{+}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_minus2['pkmu_mass'], bins=100, range=[1500, 5000])
    # plt.xlabel('$m_{p^{+}K^{-}\\mu^{-}_{\\tau}}$')
    # plt.show()
    # plt.hist(pkmu_mass_lc_plus2['pkmu_mass'], bins=100, range=[1500, 5000])
    # plt.xlabel('$m_{p^{-}K^{+}\\mu^{+}_{\\tau}}$')
    # plt.show()

    print(len(data_frame))
    print((pkmu_mass_lc_minus['pkmu_mass']).describe())
    data_frame = data_frame.drop(list((pkmu_mass_lc_minus[pkmu_mass_lc_minus['pkmu_mass'] < 2800]).index))
    data_frame = data_frame.drop(list((pkmu_mass_lc_plus[pkmu_mass_lc_plus['pkmu_mass'] < 2800]).index))
    # plt.hist(data_frame['pk_mass'], bins=100, range=[1400, 2400])
    # plt.show()

    print(len(data_frame))
    return data_frame


def analyse_pkmu_for_2_muons2(data_frame, to_plot: bool):
    """
    Analyse the pkmu mass of the data set
    :param data_frame:
    :param to_plot: if True, the pkmu mass is plotted, along with other parameters
    :return:
    """
    data_frame = data_frame.reset_index(drop=True)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pkmu_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['tauMu_P', 'mu']]
    data_frame['pkmu_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    pkmu_mass_lc_plus = data_frame[(data_frame['proton_ID'] == 2212) & (data_frame['Kminus_ID'] == -321)]
    pkmu_mass_lc_plus['pkmu_mass'] = pkmu_mass_lc_plus['pkmu_mass1']
    pkmu_mass_lc_plus['mu_id'] = pkmu_mass_lc_plus['mu1_ID']
    index_mask = list((pkmu_mass_lc_plus[pkmu_mass_lc_plus['tauMu_ID'] == -13]).index)
    pkmu_mass_lc_plus.loc[index_mask, 'pkmu_mass'] = pkmu_mass_lc_plus.loc[index_mask, 'pkmu_mass2']
    pkmu_mass_lc_plus.loc[index_mask, 'mu_id'] = pkmu_mass_lc_plus.loc[index_mask, 'tauMu_ID']
    print((pkmu_mass_lc_plus['mu_id']).describe())
    pkmu_mass_lc_minus = data_frame[(data_frame['proton_ID'] == -2212) & (data_frame['Kminus_ID'] == +321)]
    pkmu_mass_lc_minus['pkmu_mass'] = pkmu_mass_lc_minus['pkmu_mass1']
    index_mask = list((pkmu_mass_lc_minus[pkmu_mass_lc_minus['tauMu_ID'] == +13]).index)
    pkmu_mass_lc_minus.loc[index_mask, 'pkmu_mass'] = pkmu_mass_lc_minus.loc[index_mask, 'pkmu_mass2']
    plt.hist2d(pkmu_mass_lc_plus['Lb_M'], pkmu_mass_lc_plus['pkmu_mass'], bins=30, norm=LogNorm())
    plt.show()
    # plt.hist2d(pkmu_mass_lc_plus['pkmutau_mass'], pkmu_mass_lc_plus['pkmu_mass'], bins=30, norm=LogNorm(),
    #            range=[[3500, 10000], [1600, 3000]])
    # plt.show()
    plt.hist(pkmu_mass_lc_minus['pkmu_mass'], bins=100, range=[1500, 5000])
    plt.xlabel('$m_{p^{+}K^{-}\\mu^{-}}$')
    plt.show()
    plt.hist(pkmu_mass_lc_plus['pkmu_mass'], bins=100, range=[1500, 5000])
    plt.xlabel('$m_{p^{-}K^{+}\\mu^{+}}$')
    plt.show()

    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pkmu_mass1'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['tauMu_P', 'mu']]
    data_frame['pkmu_mass2'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    pkmu_mass_lc_plus1 = data_frame[(data_frame['proton_ID'] == 2212) & (data_frame['Kminus_ID'] == -321)]
    pkmu_mass_lc_plus1 = pkmu_mass_lc_plus1[(pkmu_mass_lc_plus1['mu1_ID'] == -13)]
    pkmu_mass_lc_plus1['pkmu_mass'] = pkmu_mass_lc_plus1['pkmu_mass1']
    pkmu_mass_lc_minus1 = data_frame[(data_frame['proton_ID'] == -2212) & (data_frame['Kminus_ID'] == 321)]
    pkmu_mass_lc_minus1 = pkmu_mass_lc_minus1[(pkmu_mass_lc_minus1['mu1_ID'] == +13)]
    pkmu_mass_lc_minus1['pkmu_mass'] = pkmu_mass_lc_minus1['pkmu_mass1']
    pkmu_mass_lc_plus2 = data_frame[(data_frame['proton_ID'] == 2212) & (data_frame['Kminus_ID'] == -321)]
    pkmu_mass_lc_plus2 = pkmu_mass_lc_plus2[(pkmu_mass_lc_plus2['tauMu_ID'] == -13)]
    pkmu_mass_lc_plus2['pkmu_mass'] = pkmu_mass_lc_plus2['pkmu_mass2']
    pkmu_mass_lc_minus2 = data_frame[(data_frame['proton_ID'] == -2212) & (data_frame['Kminus_ID'] == 321)]
    pkmu_mass_lc_minus2 = pkmu_mass_lc_minus2[(pkmu_mass_lc_minus2['tauMu_ID'] == +13)]
    pkmu_mass_lc_minus2['pkmu_mass'] = pkmu_mass_lc_minus2['pkmu_mass2']
    plt.hist(pkmu_mass_lc_minus1['pkmu_mass'], bins=100, range=[1500, 5000])
    plt.xlabel('$m_{p^{+}K^{-}\\mu^{-}}$')
    plt.show()
    plt.hist(pkmu_mass_lc_plus1['pkmu_mass'], bins=100, range=[1500, 5000])
    plt.xlabel('$m_{p^{-}K^{+}\\mu^{+}}$')
    plt.show()
    plt.hist(pkmu_mass_lc_minus2['pkmu_mass'], bins=100, range=[1500, 5000])
    plt.xlabel('$m_{p^{+}K^{-}\\mu^{-}_{\\tau}}$')
    plt.show()
    plt.hist(pkmu_mass_lc_plus2['pkmu_mass'], bins=100, range=[1500, 5000])
    plt.xlabel('$m_{p^{-}K^{+}\\mu^{+}_{\\tau}}$')
    plt.show()
    plt.hist(pkmu_mass_lc_minus1['tau_distances_travelled'], bins=200, range=[-10, 10])
    plt.xlabel('tau FD')
    plt.show()
    plt.hist(pkmu_mass_lc_plus1['tau_distances_travelled'], bins=200, range=[-10, 10])
    plt.xlabel('tau FD')
    plt.show()

    print(len(data_frame))
    print((pkmu_mass_lc_minus['pkmu_mass']).describe())
    # print(list((pkmu_mass_lc_minus[pkmu_mass_lc_minus['pkmu_mass'] < 2300]).index))
    data_frame = data_frame.drop(list((pkmu_mass_lc_minus[pkmu_mass_lc_minus['pkmu_mass'] < 2800]).index))
    data_frame = data_frame.drop(list((pkmu_mass_lc_plus[pkmu_mass_lc_plus['pkmu_mass'] < 2800]).index))

    print(len(data_frame))
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
        plt.hist(data_frame['kpi_mass'], bins=75, range=[500, 2500])
        plt.xlabel('$m_{K\\pi}$')
        plt.axvline(masses['kstar'], c='k')
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
    data_frame['pik_mass'] = get_mass(data_frame=data_frame,
                                      particles_associations=[['Kminus_P', 'K'], ['proton_P', 'pi']])
    data_frame['kmu_mass'] = get_mass(data_frame=data_frame,
                                      particles_associations=[['Kminus_P', 'K'], ['mu1_P', 'mu']])
    data_frame['pimu_mass'] = get_mass(data_frame=data_frame,
                                       particles_associations=[['proton_P', 'pi'], ['mu1_P', 'mu']])
    if to_plot:
        plt.hist(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]['pikmu_mass'],
                 bins=75, range=[1500, 5000])
        plt.xlabel('$m_{K\\pi\\mu}$ where pi and mu have the same charge')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]['pik_mass'],
                 bins=75, range=[500, 4000])
        plt.xlabel('$m_{K\\pi}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]['pimu_mass'],
                 bins=75, range=[400, 3500])
        plt.xlabel('$m_{\\pi\\mu}$ where p and mu have the same charge')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]['kmu_mass'],
                 bins=75, range=[500, 4000])
        plt.xlabel('$m_{K\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        df_to_plot = data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]
        plt.hist2d(df_to_plot['pikmu_mass'], df_to_plot['kmu_mass'], range=[[1500, 4500], [1000, 3000]], bins=40)
        plt.xlabel('$m_{\\pi K\\mu}$')
        plt.ylabel('$m_{K\\mu}$')
        plt.show()
        plt.hist2d(df_to_plot['pikmu_mass'], df_to_plot['kpi_mass'], range=[[1500, 4500], [1000, 3000]], bins=40)
        plt.xlabel('$m_{\\pi K\\mu}$')
        plt.ylabel('$m_{K\\pi}$')
        plt.show()
        plt.hist(df_to_plot['kmu_mass'], range=[[1500, 4500], [1000, 3000]], bins=40)
        plt.xlabel('$m_{\\pi K\\mu}$')
        plt.ylabel('$m_{K\\mu}$')
        plt.show()
    return data_frame


def plot_pikmumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pikmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi']]
    data_frame['pik_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        # data_frame = data_frame[(data_frame['Lb_M'] < 5620 - 40) | (data_frame['Lb_M'] > 5620 + 40)]
        plt.hist(data_frame['pikmumu_mass'], bins=100, range=[3500, 6500])
        plt.axvline(masses['B'], c='k')
        plt.axvline(5366, c='k')  # Bs mass
        plt.xlabel('$m_{K\\pi\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        test_part = data_frame[
            (data_frame['pikmumu_mass'] < masses['B'] + 100) & (data_frame['pikmumu_mass'] > masses['B'] - 100)]
        plt.hist(data_frame['pik_mass'], bins=50, range=[500, 2000])
        plt.axvline(masses['kstar'], c='k')
        plt.xlabel('$m_{K\\pi}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(test_part['pik_mass'], bins=100, range=[500, 2000])
        plt.axvline(masses['kstar'], c='k')
        plt.xlabel('$m_{K\\pi}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist2d(data_frame['pikmumu_mass'], data_frame['pik_mass'], bins=30, range=[[2000, 7000], [500, 2000]])
        plt.xlabel('$m_{K\\pi\\mu\\mu}$')
        plt.ylabel('$m_{K\\pi}$')
        # plt.savefig('pikmumu_pik_nolb.png')
        plt.show()
    to_remove_b = data_frame[
        ((data_frame['pikmumu_mass'] < 5366 + 200) & (data_frame['pikmumu_mass'] > masses['B'] - 200)) & (
                (data_frame['pik_mass'] > masses['kstar'] - 30) & (
                data_frame['pik_mass'] < masses['kstar'] + 30))]
    # data_frame = data_frame[
    #     (data_frame['pikmumu_mass'] > masses['B'] + 100) | (data_frame['pikmumu_mass'] < masses['B'] - 100)]
    # data_frame = data_frame[(data_frame['pikmumu_mass'] > 5366 + 100) | (data_frame['pikmumu_mass'] < 5366 - 100)]
    data_frame = data_frame.drop(to_remove_b.index)
    return data_frame


def plot_kmumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['kmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['proton_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['p(k)mumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu'], ['proton_P', 'mu']]
    data_frame['kmup(mu)_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kmumu_mass'], bins=100)
        # plt.hist(data_frame[(data_frame['Lb_M'] > 5620 - 40) & (data_frame['Lb_M'] < 5620 + 40)]['kmumu_mass'], bins=100)
        plt.xlabel('$m_{k\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['p(k)mumu_mass'], bins=100)
        # plt.hist(data_frame[(data_frame['Lb_M'] > 5620 - 40) & (data_frame['Lb_M'] < 5620 + 40)]['p(k)mumu_mass'], bins=100)
        plt.xlabel('$m_{p(k)\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['kmup(mu)_mass'], bins=100)
        # plt.hist(data_frame[(data_frame['Lb_M'] > 5620 - 40) & (data_frame['Lb_M'] < 5620 + 40)]['kmup(mu)_mass'], bins=100)
        plt.xlabel('$m_{k\\mu p(\\mu )}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def plot_kpmumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['kpmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'K']]
    data_frame['kp_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kpmumu_mass'], bins=100, range=[3500, 8000])
        plt.hist(data_frame[(data_frame['Lb_M'] > 5620 - 40) & (data_frame['Lb_M'] < 5620 + 40)]['kpmumu_mass'],
                 bins=100, range=[3500, 8000])
        plt.axvline(masses['Lb'], c='k')
        plt.xlabel('$m_{kp\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame[(data_frame['Lb_M'] < 5620 - 40) | (data_frame['Lb_M'] > 5620 + 40)]['kpmumu_mass'],
                 bins=100, range=[3500, 8000])
        plt.axvline(masses['Lb'], c='k')
        plt.xlabel('$m_{kp\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist2d(data_frame['kpmumu_mass'], data_frame['kp_mass'], bins=50, range=[[3500, 6500], [1450, 3000]])
        plt.show()
    data_frame = data_frame[(data_frame['kpmumu_mass'] < 5620 - 40) | (data_frame['kpmumu_mass'] > 5620 + 40)]
    return data_frame


def plot_ppimumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['ppimumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'proton']]
    data_frame['ppi_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        test_data = data_frame[(data_frame['ppimumu_mass'] > 5400) & (data_frame['ppimumu_mass'] < 5600)]
        plt.hist(data_frame['ppimumu_mass'], bins=150, range=[4000, 7000])
        plt.hist(data_frame[(data_frame['Lb_M'] > 5620 - 40) & (data_frame['Lb_M'] < 5620 + 40)]['ppimumu_mass'],
                 bins=150, range=[3000, 7000])
        plt.axvline(masses['Lb'], c='k')
        plt.xlabel('$m_{p\\pi\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['ppi_mass'], bins=100, range=[1000, 3000])
        plt.xlabel('$m_{p\\pi}$')
        plt.ylabel('occurrences')
        plt.axvline(1115, c='k')
        plt.show()
        plt.hist(test_data['ppi_mass'], bins=100, range=[1000, 3000])
        plt.xlabel('$m_{p\\pi}$')
        plt.ylabel('occurrences')
        plt.axvline(1115, c='k')
        plt.show()
        plt.hist2d(data_frame['ppimumu_mass'], data_frame['ppi_mass'], bins=30, range=[[3000, 7000], [1000, 4000]])
        plt.xlabel('$m_{p\\pi\\mu\\mu}$')
        plt.ylabel('$m_{p\\pi}$')
        plt.axvline(masses['Lb'], c='k')
        plt.axhline(1115, c='k')
        plt.show()
        plt.hist2d(data_frame['ppimumu_mass'], data_frame['Lb_M'], bins=60, range=[[3000, 7000], [3000, 7000]])
        plt.xlabel('$m_{p\\pi\\mu\\mu}$')
        plt.ylabel('$m_{Lb}$')
        plt.show()
    data_frame = data_frame[(data_frame['ppimumu_mass'] < 5420) | (data_frame['ppimumu_mass'] > 5620)]
    return data_frame


def plot_pmumumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'mu'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pmumumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'mu'], ['proton_P', 'proton']]
    data_frame['pmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['pmumumu_mass'], bins=150, range=[3000, 7000])
        plt.axvline(masses['Lb'], c='k')
        plt.xlabel('$m_{p\\mu\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['pmu_mass'], bins=100, range=[1000, 4000])
        # plt.axvline(masses['Lb'])
        plt.xlabel('$m_{p\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist2d(data_frame['pmumumu_mass'], data_frame['pmu_mass'], bins=30, range=[[3000, 7000], [1000, 4000]])
        plt.xlabel('$m_{p\\mu\\mu\\mu}$')
        plt.ylabel('$m_{p\\mu}$')
        # plt.axvline(masses['Lb'])
        plt.show()
    return data_frame


def plot_pipimumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'pi'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pipimumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'pi']]
    data_frame['pipi_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['mumu_mass'] = get_mass(data_frame=data_frame,
                                       particles_associations=[['mu1_P', 'mu'], ['tauMu_P', 'mu']])
    # data_frame['pmu1_mass'] = get_mass(data_frame=data_frame, particles_associations=[['proton_P', 'pi'], ['mu1_P', 'mu']])
    # data_frame['pmu2_mass'] = get_mass(data_frame=data_frame, particles_associations=[['proton_P', 'pi'], ['tauMu_P', 'mu']])
    # data_frame['kmu1_mass'] = get_mass(data_frame=data_frame, particles_associations=[['Kminus_P', 'pi'], ['mu1_P', 'mu']])
    # data_frame['kmu2_mass'] = get_mass(data_frame=data_frame, particles_associations=[['Kminus_P', 'pi'], ['tauMu_P', 'mu']])
    # combo1 = data_frame[np.sign(data_frame['proton_ID']) == np.sign(data_frame['mu1_ID'])]
    # combo2 = data_frame[np.sign(data_frame['proton_ID']) == np.sign(data_frame['tauMu_ID'])]
    # combo3 = data_frame[np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['mu1_ID'])]
    # combo4 = data_frame[np.sign(data_frame['Kminus_ID']) == np.sign(data_frame['tauMu_ID'])]
    data_frame['pipipi1_mass'] = get_mass(data_frame=data_frame,
                                          particles_associations=[['proton_P', 'pi'], ['mu1_P', 'mu'],
                                                                  ['tauMu_P', 'mu']])
    data_frame['pipipi2_mass'] = get_mass(data_frame=data_frame,
                                          particles_associations=[['Kminus_P', 'pi'], ['mu1_P', 'mu'],
                                                                  ['tauMu_P', 'mu']])
    data_frame['pipipi3_mass'] = get_mass(data_frame=data_frame,
                                          particles_associations=[['Kminus_P', 'pi'], ['proton_P', 'pi'],
                                                                  ['mu1_P', 'mu']])
    data_frame['pipipi4_mass'] = get_mass(data_frame=data_frame,
                                          particles_associations=[['Kminus_P', 'pi'], ['proton_P', 'pi'],
                                                                  ['tauMu_P', 'mu']])
    if to_plot:
        plt.hist2d(data_frame['pipimumu_mass'], data_frame['pipi_mass'], bins=30, range=[[2000, 7000], [200, 2000]])
        plt.xlabel('$m_{\\pi\\pi\\mu\\mu}$')
        plt.ylabel('$m_{\\pi\\pi}$')
        plt.show()
        # data_frame = data_frame[(data_frame['Lb_M'] < 5620 - 40) | (data_frame['Lb_M'] > 5620 + 40)]
        # plt.hist2d(data_frame['pipimumu_mass'], data_frame['Lb_M'], bins=30, range=[[3500, 6000], [4500, 6500]])
        # plt.xlabel('$m_{\\pi\\pi\\mu\\mu}$')
        # plt.ylabel('$m_{Lb}$')
        plt.show()
        # plt.hist(combo1['pmu1_mass'], bins=100, range=[200, 2500])
        # plt.xlabel('combo 1')
        # plt.show()
        # plt.hist(combo2['pmu2_mass'], bins=100, range=[200, 2500])
        # plt.xlabel('combo 2')
        # plt.show()
        # plt.hist(combo3['kmu1_mass'], bins=100, range=[200, 2500])
        # plt.xlabel('combo 3')
        # plt.show()
        # plt.hist(combo4['kmu2_mass'], bins=100, range=[200, 2500])
        # plt.xlabel('combo 4')
        # plt.show()
        # for i in [1, 2, 3, 4]:
        #     plt.hist(data_frame[f'pipipi{i}_mass'], bins=100, range=[500, 6000])
        #     plt.xlabel('$m_{\\pi\\pi\\pi} $' + str(i))
        #     plt.ylabel('occurrences')
        #     plt.axvline(masses['B'], c='k')
        #     plt.show()
        plt.hist(data_frame['pipimumu_mass'], bins=100, range=[2000, 7000])
        plt.xlabel('$m_{\\pi\\pi\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.axvline(masses['B'], c='k')
        plt.show()
        test_data = data_frame[
            (data_frame['pipimumu_mass'] > masses['B']) & (data_frame['pipimumu_mass'] < masses['B'] + 100)]
        test_data = test_data[
            (data_frame['pipi_mass'] > masses['K'] - 100) & (data_frame['pipi_mass'] < masses['K'] + 100)]
        # test_data = data_frame[(data_frame['pipi_mass'] < masses['K'] + 100)]
        # for i in [1, 2, 3, 4]:
        #     plt.hist(test_data[f'pipipi{i}_mass'], bins=100, range=[500, 6000])
        #     plt.xlabel('$m_{\\pi\\pi pi }$' + str(i))
        #     plt.ylabel('occurrences')
        #     plt.show()
        plt.hist(test_data['pipimumu_mass'], bins=100, range=[2000, 7000])
        plt.xlabel('$m_{\\pi\\pi\\mu\\mu}$ test data')
        plt.ylabel('occurrences')
        plt.axvline(masses['B'], c='k')
        plt.show()
        plt.hist(test_data['Lb_M'], bins=100, range=[2000, 7000])
        plt.xlabel('$m_{Lb}$ test data')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['pipi_mass'], bins=100, range=[200, 2000])
        plt.xlabel('$m_{\\pi\\pi}$')
        plt.ylabel('occurrences')
        plt.axvline(masses['K'], c='k')
        plt.show()
        plt.hist(data_frame['mumu_mass'], bins=100, range=[200, 2000])
        plt.xlabel('$m_{\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.axvline(masses['K'], c='k')
        plt.show()
        plt.hist(test_data['pipi_mass'], bins=100, range=[200, 2000])
        plt.xlabel('$m_{\\pi\\pi}$ test data')
        plt.ylabel('occurrences')
        plt.axvline(masses['K'], c='k')
        plt.show()
    data_frame = data_frame[
        (data_frame['pipimumu_mass'] < masses['B']) | (data_frame['pipimumu_mass'] > masses['B'] + 100)]
    return data_frame


def plot_kkmumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['kkmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'K']]
    data_frame['kk_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        # data_frame = data_frame[(data_frame['Lb_M'] < 5620 - 40) | (data_frame['Lb_M'] > 5620 + 40)]
        plt.hist(data_frame['kkmumu_mass'], bins=100, range=[2500, 7000])
        plt.xlabel('$m_{KK\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.axvline(5366, c='k')  # Bs mass
        plt.show()
        test_data = data_frame[(data_frame['kkmumu_mass'] > 5366 - 100) & (data_frame['kkmumu_mass'] < 5366 + 100)]
        # test_data = data_frame[data_frame['kk_mass'] < 1120]
        plt.hist(data_frame['kk_mass'], bins=100, range=[900, 2000])
        plt.xlabel('$m_{KK}$')
        plt.ylabel('occurrences')
        plt.axvline(1020, c='k')
        plt.show()
        plt.hist(test_data['kk_mass'], bins=100, range=[900, 2000])
        plt.xlim(right=2000)
        plt.xlabel('$m_{KK}$')
        plt.ylabel('occurrences')
        plt.axvline(1020, c='k')
        plt.show()
        plt.hist(test_data['kkmumu_mass'], bins=100)
        plt.xlim(right=7000)
        plt.xlabel('$m_{KK\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.axvline(5366, c='k')
        plt.show()
        plt.hist2d(data_frame['kkmumu_mass'], data_frame['kk_mass'], bins=30, range=[[2000, 7000], [900, 2000]])
        plt.axvline(5366, c='k')
        plt.axhline(1020, c='k')
        plt.xlabel('$m_{KK\\mu\\mu}$')
        plt.ylabel('$m_{KK}$')
        plt.show()
    to_drop = data_frame[((data_frame['kkmumu_mass'] > 5366 - 200) & (data_frame['kkmumu_mass'] < 5366 + 200)) & (
            (data_frame['kk_mass'] > 990) & (data_frame['kk_mass'] < 1050))]
    data_frame = data_frame.drop(to_drop.index)
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
        plt.hist(data_frame['pmumu_mass'], bins=100)
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
    df_to_plot = data_frame[np.sign(data_frame['proton_ID']) == np.sign(data_frame['mu1_ID'])]
    plt.hist(df_to_plot['pmu_mass'], bins=100, range=[1400, 3700])
    plt.xlim(right=3700)
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
    data_frame = data_frame[(data_frame['pk_mass'] < 1500) | (data_frame['pk_mass'] > 1540)]
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
    data_frame['dimuon_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    compare_data = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3050)]
    print('jpsi events', len(compare_data))
    compare_data = compare_data[(compare_data['Lb_M'] < 5650) & (compare_data['Lb_M'] > 5590)]
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


def jpsi_swaps(data_frame):
    """
    Look for jpsi swaps
    :param data_frame:
    :return:
    """
    particles_duos = [['mu1_P', 'tauMu_P'], ['mu1_P', 'proton_P'], ['mu1_P', 'Kminus_P'], ['tauMu_P', 'proton_P'],
                      ['tauMu_P', 'Kminus_P'], ['proton_P', 'Kminus_P']]
    for p1, p2 in particles_duos:
        particles_associations = [[p1, 'mu'], [p2, 'mu']]
        if ('mu' in p1.lower() and 'mu' in p2.lower()) or ('mu' not in p1.lower() and 'mu' not in p2.lower()):
            mass_frame = data_frame[np.sign(data_frame[p1[:-2] + '_ID']) != np.sign(data_frame[p2[:-2] + '_ID'])]
        else:
            mass_frame = data_frame[np.sign(data_frame[p1[:-2] + '_ID']) == np.sign(data_frame[p2[:-2] + '_ID'])]
        mass = get_mass(data_frame=mass_frame, particles_associations=particles_associations)
        plt.hist(mass, bins=100)
        plt.axvline(masses['J/psi'], c='k')
        plt.xlabel(p1[:-2] + ' ' + p2[:-2])
        plt.xlim(right=4500)
        plt.show()
    return data_frame


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
