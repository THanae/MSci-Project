import matplotlib.pyplot as plt

from background_reduction.background_reduction_methods import clean_cuts, identify_p_k_j_psi, pid_cleaning, \
    impact_parameter_cleaning, chi2_cleaning
from data_loader import load_data, add_branches
from ip_calculations import line_point_distance
from plotting_functions import plot_columns, plot_compare_data


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

    data_frame = identify_p_k_j_psi(data_frame, False)
    return data_frame


if __name__ == '__main__':
    a = load_data(add_branches())
    data_frame = b_cleaning(a)
    # data_frame = clean_cuts(data_frame)
    # print('cuts cleaned', len(data_frame))
    data_frame = identify_p_k_j_psi(data_frame, False)
    print('j/psi cleaning', len(data_frame))
    # data_frame = pid_cleaning(data_frame)
    print('PID cleaning', len(data_frame))
    # data_frame = impact_parameter_cleaning(data_frame)
    # print('impact parameter cleaning', len(data_frame))
    # data_frame = chi2_cleaning(data_frame)
    # print('chi squared cleaning', len(data_frame))
    # data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT1'] < -0]
    # print('isolation angle cleaning', len(data_frame))
    # data_frame = data_frame.reset_index(drop=True)
    # data_frame['vector_muTau'] = data_frame[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    # data_frame['tauMu_reference_point'] = data_frame[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    # data_frame['pkmu_endvertex_point'] = data_frame[['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    # data_frame['impact_parameter_thingy'] = line_point_distance(vector=data_frame['vector_muTau'],
    #                                                             vector_point=data_frame['tauMu_reference_point'],
    #                                                             point=data_frame['pkmu_endvertex_point'])
    # data_frame = data_frame[data_frame['impact_parameter_thingy'] > -0.02]
    data_frame = data_frame.reset_index(drop=True)
    print(len(data_frame))
