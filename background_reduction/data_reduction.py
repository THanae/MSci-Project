from background_reduction.background_reduction_methods import clean_cuts, identify_p_k_j_psi, impact_parameter_cleaning, \
    pid_cleaning, chi2_cleaning, analyse_pkmu_for_2_muons, plot_pipimumu_mass, plot_ppimumu_mass, plot_pikmumu_mass, \
    kpi_mass, plot_pikmu_mass, plot_pk_mass, pp_mass, kmu_mass, kk_mass, pmu_mass, plot_pmu_mass, \
    plot_kmu_mass, plot_pmumumu_mass, plot_kmumu_mass, jpsi_swaps, plot_kkmumu_mass, plot_pmumu_mass, plot_kpmumu_mass, \
    ip_star_cleaning
from data_loader import load_data, add_branches
from ip_calculations import line_point_distance


def reduce_background(data_frame, bdt=False):
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
        # data_frame = data_frame[data_frame['pKmu_ENDVERTEX_CHI2'] < 15]
        # data_frame = data_frame[data_frame['Lb_FDCHI2_OWNPV'] > 300]
        print('chi squared cleaning', len(data_frame))
        data_frame = data_frame[data_frame['Lb_pmu_ISOLATION_BDT1'] < -0]
        print('isolation angle cleaning', len(data_frame))
        # data_frame = transverse_momentum_cleaning(data_frame, False)
        print('transverse momentum cleaning', len(data_frame))
    data_frame = analyse_pkmu_for_2_muons(data_frame, True)
    print('Lc cleaning', len(data_frame))
    if not bdt:
        data_frame = data_frame.reset_index(drop=True)
        data_frame = ip_star_cleaning(data_frame)
        print('ip star cleaning', len(data_frame))
    # data_frame = plot_kpmumu_mass(data_frame, False)
    # tests to check for further stuff
    # data_frame = plot_ppimumu_mass(data_frame, False)
    # data_frame = plot_kkmumu_mass(data_frame, False)
    # data_frame = plot_pipimumu_mass(data_frame, False)
    # data_frame = plot_pikmumu_mass(data_frame, False)
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


if __name__ == '__main__':
    a = load_data(add_branches())
    a.dropna(inplace=True)
    # plot_columns(a, [-75, 75], ['Lb_pmu_TR1_PIDp', 'Lb_pmu_TR1_PIDK', 'Lb_pmu_TR1_PIDmu', 'Lb_pmu_TR1_PIDpi'], 100)
    df = reduce_background(a)

    df['vector_muTau'] = df[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    df['tauMu_reference_point'] = df[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    df['pkmu_endvertex_point'] = df[['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    df['pkmu_direction'] = df[['pKmu_PX', 'pKmu_PY', 'pKmu_PZ']].values.tolist()
    df['impact_parameter_thingy'] = line_point_distance(vector=df['vector_muTau'],
                                                        vector_point=df['tauMu_reference_point'],
                                                        point=df['pkmu_endvertex_point'],
                                                        direction=df['pkmu_direction'])
    plot_kpmumu_mass(df, True)
    # df = df[(df['Lb_M'] < 5620 - 40) | (df['Lb_M'] > 5620 + 40)]
    # df = df[df['impact_parameter_thingy'] > -0.02]
    # x = df['proton_PX']
    # y = df['proton_PY']
    # z = df['proton_PZ']
    # p = df['proton_P']
    # a = p**2 - x**2 - y**2 - z**2
    # print(a.describe())
    plot_kmumu_mass(df, True)
    plot_ppimumu_mass(df, True)
    plot_kpmumu_mass(df, True)
    # analyse_pkmu_for_2_muons(df, True)
    # print(len(df))
    # plot_kkmumu_mass(df, True)
    # jpsi_swaps(df)
    # plot_pmu_mass(df, to_plot=True)
    # plot_kmumu_mass(df, True)
    plot_pikmu_mass(df, True)
    # plot_pmumumu_mass(df, True)
    # plot_ppimumu_mass(df, True)
    # plot_pipimumu_mass(df, True)
    # plot_ppimumu_mass(df, True)
    plot_pikmumu_mass(df, True)

    # kpi_mass(df, True)
    plot_pikmu_mass(df, True)
    # df = plot_pk_mass(df)
    pp_mass(data_frame=df, to_plot=True)
    kpi_mass(df, True)
    kmu_mass(df, True)
    kk_mass(df, True)
    pmu_mass(df, True)

