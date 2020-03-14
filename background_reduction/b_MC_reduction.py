import matplotlib.pyplot as plt
import numpy as np

from masses import masses, get_mass
from background_reduction.background_reduction_methods import identify_p_k_j_psi
from data.data_loader import load_data
from plotting_functions import plot_columns, plot_compare_data


def b_cleaning(data_frame, to_plot=False, pkmu_threshold: int = 2800):
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

    data_frame = identify_p_k_j_psi(data_frame, False)
    # data_frame = kmu_cut(data_frame)
    # print('Kmu cleaning', len(data_frame))
    data_frame['stretched'] = get_stretched_pikmu_mass(data_frame,
                                                       [['proton_P', 'pi'], ['Kminus_P', 'K'], ['mu1_P', 'mu']])
    data_frame['stretched2'] = get_stretched_pikmu_mass(data_frame,
                                                        [['proton_P', 'pi'], ['Kminus_P', 'K'], ['tauMu_P', 'mu']])
    df1 = data_frame[np.sign(data_frame['proton_TRUEID']) != np.sign(data_frame['mu1_TRUEID'])]
    df2 = data_frame[np.sign(data_frame['proton_TRUEID']) == np.sign(data_frame['mu1_TRUEID'])]
    n = len(df1) + len(df2)
    print(n)
    print(len(df1[df1['stretched'] > 2800]) / len(df1))
    print(len(df1[df1['stretched'] > 3000]) / len(df1))
    print(len(df2[df2['stretched2'] > 2800]) / len(df2))
    print(len(df2[df2['stretched2'] > 3000]) / len(df2))
    print((len(df2[df2['stretched2'] > 2800]) + len(df1[df1['stretched'] > 2800])) / n)
    to_drop_1, to_drop_2 = df1[df1['stretched'] < pkmu_threshold], df2[df2['stretched2'] < pkmu_threshold]
    data_frame = data_frame.drop(list(to_drop_1.index))
    data_frame = data_frame.drop(list(to_drop_2.index))
    data_frame = data_frame.reset_index(drop=True)
    proton_P_threshold, proton_PT_threshold = 15e3, 1000
    mu1_P_threshold, mu1_PT_threshold = 10e3, 1500
    tauMu_P_threshold, tauMu_PT_threshold = 10e3, 1500
    data_frame = data_frame[data_frame['proton_P'] > proton_P_threshold]
    data_frame = data_frame[data_frame['proton_PT'] > proton_PT_threshold]
    data_frame = data_frame[data_frame['mu1_P'] > mu1_P_threshold]
    data_frame = data_frame[data_frame['mu1_PT'] > mu1_PT_threshold]
    data_frame = data_frame[data_frame['tauMu_P'] > tauMu_P_threshold]
    data_frame = data_frame[data_frame['tauMu_PT'] > tauMu_PT_threshold]
    # data_frame['pikmu_mass'] = get_mass(data_frame, [['proton_P', 'pi'], ['Kminus_P', 'K'], ['mu1_P', 'mu']])
    # print(len(data_frame))
    # data_frame = data_frame[data_frame['pikmu_mass'] < masses['B'] - masses['tau']]
    # print(len(data_frame))
    data_frame['stretchedkmu'] = get_stretched_kmu_mass(data_frame, [['Kminus_P', 'K'], ['mu1_P', 'mu']])
    data_frame['stretchedkmu2'] = get_stretched_kmu_mass(data_frame, [['Kminus_P', 'K'], ['tauMu_P', 'mu']])
    df1_kmu = data_frame[np.sign(data_frame['proton_TRUEID']) != np.sign(data_frame['mu1_TRUEID'])]
    df2_kmu = data_frame[np.sign(data_frame['proton_TRUEID']) == np.sign(data_frame['mu1_TRUEID'])]
    n = len(df1_kmu) + len(df1_kmu)
    print(len(df1_kmu[df1_kmu['stretchedkmu'] > masses['D0']]) / len(df1_kmu))
    print(len(df2_kmu[df2_kmu['stretchedkmu2'] > masses['D0']]) / len(df2_kmu))
    print((len(df2_kmu[df2_kmu['stretchedkmu2'] > masses['D0']]) + len(df1_kmu[df1_kmu['stretchedkmu'] > masses['D0']])) / n)
    to_drop_1_kmu, to_drop_2_kmu = df1_kmu[df1_kmu['stretchedkmu'] < masses['D0']], df2_kmu[df2_kmu['stretchedkmu2'] < masses['D0']]
    data_frame = data_frame.drop(list(to_drop_1_kmu.index))
    data_frame = data_frame.drop(list(to_drop_2_kmu.index))
    data_frame = data_frame.reset_index(drop=True)
    print(len(data_frame))
    return data_frame


def get_stretched_pikmu_mass(data_frame, particles_associations):
    """
    Obtains (here) distribution of the pkmu mass from the pikmu mass
    :param data_frame:
    :param particles_associations:
    :return:
    """
    sum_m = get_mass(data_frame, particles_associations=particles_associations)
    sum_m = sum_m - 739  # supposed minimum mass of pikmu
    sum_m = sum_m / (3502 - 739)  # supposed to be range covered by pikmu mass
    sum_m = sum_m * (3843 - 1537)  # supposed to be range covered by pkmu mass
    sum_m = sum_m + 1537  # supposed to be minimum mass of pkmu
    return sum_m


def get_stretched_kmu_mass(data_frame, particles_associations):
    """
    Obtains (here) distribution of the kmu mass from the B MC kmu mass
    :param data_frame:
    :param particles_associations:
    :return:
    """
    sum_m = get_mass(data_frame, particles_associations=particles_associations)
    minimum_kmu_mass = masses['K'] + masses['mu']
    maximum_kmu_mass_b = masses['B'] - masses['pi'] - masses['tau']
    maximum_kmu_mass_lb = masses['Lb'] - masses['proton'] - masses['tau']
    sum_m = sum_m - minimum_kmu_mass
    sum_m = sum_m / (maximum_kmu_mass_b - minimum_kmu_mass)
    sum_m = sum_m * (maximum_kmu_mass_lb - minimum_kmu_mass)
    sum_m = sum_m + minimum_kmu_mass
    return sum_m


def diff_candidates_check(data_frame):
    uniques_p = np.unique(data_frame[['proton_TRACK_Key', 'proton_MC_MOTHER_KEY', 'proton_MC_GD_MOTHER_KEY',
                                      'proton_MC_GD_GD_MOTHER_KEY']].values.tolist(), axis=0)
    uniques_k = np.unique(data_frame[['Kminus_TRACK_Key', 'Kminus_MC_MOTHER_KEY', 'Kminus_MC_GD_MOTHER_KEY',
                                      'Kminus_MC_GD_GD_MOTHER_KEY']].values.tolist(), axis=0)
    uniques_mu1 = np.unique(data_frame[['mu1_TRACK_Key', 'mu1_MC_MOTHER_KEY', 'mu1_MC_GD_MOTHER_KEY',
                                        'mu1_MC_GD_GD_MOTHER_KEY']].values.tolist(), axis=0)
    uniques_taumu = np.unique(data_frame[['tauMu_TRACK_Key', 'tauMu_MC_MOTHER_KEY', 'tauMu_MC_GD_MOTHER_KEY',
                                          'tauMu_MC_GD_GD_MOTHER_KEY']].values.tolist(), axis=0)
    uniques_events = np.unique(data_frame[['proton_TRACK_Key', 'Kminus_TRACK_Key', 'mu1_TRACK_Key', 'tauMu_TRACK_Key',
                                           'proton_MC_MOTHER_KEY', 'Kminus_MC_MOTHER_KEY', 'mu1_MC_MOTHER_KEY',
                                           'tauMu_MC_MOTHER_KEY',
                                           'proton_MC_GD_MOTHER_KEY', 'Kminus_MC_GD_MOTHER_KEY', 'mu1_MC_GD_MOTHER_KEY',
                                           'tauMu_MC_GD_MOTHER_KEY',
                                           'proton_MC_GD_GD_MOTHER_KEY', 'Kminus_MC_GD_GD_MOTHER_KEY',
                                           'mu1_MC_GD_GD_MOTHER_KEY', 'tauMu_MC_GD_GD_MOTHER_KEY']].values.tolist(),
                               axis=0)
    print(len(uniques_mu1), len(uniques_k), len(uniques_p), len(uniques_taumu), len(uniques_events), len(data_frame))


if __name__ == '__main__':
    a = load_data(df_name='B_MC')
    df = b_cleaning(a)
    diff_candidates_check(df)
    df['stretched'] = get_stretched_pikmu_mass(df, [['proton_P', 'pi'], ['Kminus_P', 'K'], ['mu1_P', 'mu']])
    df['stretched2'] = get_stretched_pikmu_mass(df, [['proton_P', 'pi'], ['Kminus_P', 'K'], ['tauMu_P', 'mu']])
    df1 = df[np.sign(df['proton_TRUEID']) != np.sign(df['mu1_TRUEID'])]
    df2 = df[np.sign(df['proton_TRUEID']) == np.sign(df['mu1_TRUEID'])]
    n = len(df1) + len(df2)
    print(n)
    print(len(df1[df1['stretched'] > 2800]) / len(df1))
    print(len(df1[df1['stretched'] > 3000]) / len(df1))
    print(len(df2[df2['stretched2'] > 2800]) / len(df2))
    print(len(df2[df2['stretched2'] > 3000]) / len(df2))
    print((len(df2[df2['stretched2'] > 2800]) + len(df1[df1['stretched'] > 2800])) / n)
    print((len(df2[df2['stretched2'] > 3000]) + len(df1[df1['stretched'] > 3000])) / n)
    plt.hist(df1['stretched'], bins=100)
    plt.xlim(right=4000)
    plt.xlabel('pikmu stretched')
    plt.axvline(2800, c='k')
    plt.axvline(3000, c='k')
    plt.show()
    df['stretched'] = get_stretched_kmu_mass(df, [['Kminus_P', 'K'], ['mu1_P', 'mu']])
    df['stretched2'] = get_stretched_kmu_mass(df, [['Kminus_P', 'K'], ['tauMu_P', 'mu']])
    df1 = df[np.sign(df['proton_TRUEID']) != np.sign(df['mu1_TRUEID'])]
    df2 = df[np.sign(df['proton_TRUEID']) == np.sign(df['mu1_TRUEID'])]
    n = len(df1) + len(df2)
    print(len(df1[df1['stretched'] > 1850]) / len(df1))
    print(len(df2[df2['stretched2'] > 1850]) / len(df2))
    print((len(df2[df2['stretched2'] > 1850]) + len(df1[df1['stretched'] > 1850])) / n)
    plt.hist(df1['stretched'], bins=100)
    plt.xlim(right=3000)
    plt.xlabel('kmu stretched')
    plt.axvline(1850, c='k')
    plt.show()

    # df = clean_cuts(df)
    # print('cuts cleaned', len(df))
    df = identify_p_k_j_psi(df, False)
    print('j/psi cleaning', len(df))
    # df = pid_cleaning(df)
    print('PID cleaning', len(df))
    # df = impact_parameter_cleaning(df)
    # print('impact parameter cleaning', len(df))
    # df = chi2_cleaning(df)
    # print('chi squared cleaning', len(df))
    # df = df[df['Lb_pmu_ISOLATION_BDT1'] < -0]
    # print('isolation angle cleaning', len(df))
    # df = df.reset_index(drop=True)
    # df['vector_muTau'] = df[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    # df['tauMu_reference_point'] = df[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    # df['pkmu_endvertex_point'] = df[['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    # df['impact_parameter_thingy'] = line_point_distance(vector=df['vector_muTau'],
    #                                                             vector_point=df['tauMu_reference_point'],
    #                                                             point=df['pkmu_endvertex_point'])
    # df = df[df['impact_parameter_thingy'] > -0.02]
    df = df.reset_index(drop=True)
    print(len(df))
