from background_reduction.background_reduction_methods import clean_cuts, identify_p_k_j_psi, \
    pid_cleaning, chi2_cleaning, analyse_pkmu_for_2_muons, ip_star_cleaning, kmu_cut, remove_high_pkmu_mass, \
    impact_parameter_cleaning
from background_reduction.mass_replacements import plot_pipimumu_mass, plot_ppimumu_mass, plot_pikmumu_mass, \
    plot_pikmu_mass, plot_pk_mass, pp_mass, kk_mass, plot_pmu_mass, \
    plot_kmumu_mass, jpsi_swaps, plot_kkmumu_mass, plot_pmumu_mass, plot_kpmumu_mass
from data_loader import load_data


def reduce_background(data_frame, bdt=False, pkmu_threshold: int=2800):
    data_frame = clean_cuts(data_frame)
    print('cuts cleaned', len(data_frame))
    data_frame = identify_p_k_j_psi(data_frame, False)
    print('j/psi cleaning', len(data_frame))
    data_frame = pid_cleaning(data_frame)
    print('PID cleaning', len(data_frame))
    data_frame = data_frame[data_frame['mu1_isMuon']]
    print('mu1_isMuon cleaning', len(data_frame))
    data_frame = data_frame[data_frame['tauMu_isMuon']]
    print('tauMu_isMuon cleaning', len(data_frame))
    if not bdt:
        data_frame = impact_parameter_cleaning(data_frame)
        print('impact parameter cleaning', len(data_frame))
        data_frame = chi2_cleaning(data_frame)
        print('chi squared cleaning', len(data_frame))
        data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT1'] < -0]
        print('isolation angle cleaning', len(data_frame))
        # data_frame = transverse_momentum_cleaning(data_frame, False)
        # print('transverse momentum cleaning', len(data_frame))
    data_frame = analyse_pkmu_for_2_muons(data_frame, to_plot=False, pkmu_threshold=pkmu_threshold)
    print('Lc cleaning', len(data_frame))
    # data_frame = kmu_cut(data_frame)
    print('Kmu cleaning', len(data_frame))
    data_frame = remove_high_pkmu_mass(data_frame)
    print('high pkmu cleaning', len(data_frame))
    if not bdt:
        data_frame = data_frame.reset_index(drop=True)
        data_frame = ip_star_cleaning(data_frame)
        print('ip star cleaning', len(data_frame))
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


if __name__ == '__main__':
    a = load_data(df_name='Lb_data')
    a.dropna(inplace=True)
    df = reduce_background(a)
    # df = a
    # df = df[df['proton_P'] < 40000]
    # df = df[df['Kminus_P'] < 60000]
    # plt.hist(df['proton_P'], bins=50)
    # plt.xlabel('proton_P')
    # plt.show()
    # plt.hist(df['Kminus_P'], bins=50)
    # plt.xlabel('Kminus_P')
    # plt.show()
    # plt.hist2d(df['proton_P'], df['Kminus_P'], bins=75)
    # plt.xlabel('proton_P')
    # plt.ylabel('Kminus_P')
    # plt.show()
    plot_kpmumu_mass(df, True)
    # df = df[(df['Lb_M'] < 5620 - 40) | (df['Lb_M'] > 5620 + 40)]
    plot_kmumu_mass(df, True)
    plot_ppimumu_mass(df, True)
    # plot_kkmumu_mass(df, True)
    # jpsi_swaps(df)
    plot_pikmu_mass(df, True)
    # plot_pipimumu_mass(df, True)
    plot_pikmumu_mass(df, True)
    # plot_pk_mass(df)
    pp_mass(data_frame=df, to_plot=True)
    kk_mass(df, True)
