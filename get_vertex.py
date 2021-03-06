from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from matplotlib.colors import LogNorm
import seaborn as sns

from ip_calculations import line_point_distance, return_all_ip
from masses import masses, get_mass
from matrix_calculations import find_determinant_of_dir_matrix, find_inverse_of_dir_matrix


def obtain_lb_line_of_flight(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Obtains the Lambda B line of flight and flight distance
    :param data_frame:
    :return:
    """
    all_distances, vectors, errors = [], [], []
    print(f'data frame length: {len(data_frame)}, FD 90th percentile: {np.percentile(data_frame["Lb_FD_OWNPV"], 90)}')
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        pv_xyz = [ts['Lb_OWNPV_X'], ts['Lb_OWNPV_Y'], ts['Lb_OWNPV_Z']]
        end_xyz = [ts['pKmu_ENDVERTEX_X'], ts['pKmu_ENDVERTEX_Y'], ts['pKmu_ENDVERTEX_Z']]
        distance = np.linalg.norm(np.array(pv_xyz) - np.array(end_xyz))
        vector = np.array(end_xyz) - np.array(pv_xyz)
        vectors.append(vector)
        all_distances.append(distance)
    data_frame['lb_distances'] = all_distances
    data_frame['vectors'] = vectors
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def line_plane_intersection(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Finds intersection between the tauMu, and the plane made of the Lb line of light and pkmu momentum vector
    :param data_frame:
    :return:
    """
    intersections, muon_from_tau = [], []
    angles_mu = []
    calc, dets = [], []
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        end_xyz = [ts['pKmu_ENDVERTEX_X'], ts['pKmu_ENDVERTEX_Y'], ts['pKmu_ENDVERTEX_Z']]
        momentum_pkmu = [ts['pKmu_PX'], ts['pKmu_PY'], ts['pKmu_PZ']]
        momentum_tauMu = [ts['tauMu_PX'], ts['tauMu_PY'], ts['tauMu_PZ']]
        point_tau_mu = [ts['tauMu_REFPX'], ts['tauMu_REFPY'], ts['tauMu_REFPZ']]
        vector_plane_lb = ts['vectors']
        coefficient_matrix = [[vector_plane_lb[i], momentum_pkmu[i], - momentum_tauMu[i]] for i in range(3)]
        determinant = find_determinant_of_dir_matrix(vector_plane_lb=vector_plane_lb, momentum_pkmu=momentum_pkmu,
                                                     momentum_tauMu=momentum_tauMu)
        inverse_matrix = find_inverse_of_dir_matrix(vector_plane_lb=vector_plane_lb, momentum_pkmu=momentum_pkmu,
                                                    momentum_tauMu=momentum_tauMu, determinant=determinant)
        if np.linalg.matrix_rank(coefficient_matrix) == np.array(coefficient_matrix).shape[0]:
            ordinate = [point_tau_mu[j] - end_xyz[j] for j in range(3)]
            possible_calculation_u = sum([inverse_matrix[2][j] * ordinate[j] for j in range(3)])
            possible_intersection = np.array(momentum_tauMu) * possible_calculation_u + np.array(point_tau_mu)
        else:
            possible_intersection = False
        if possible_intersection is not False:
            intersection = possible_intersection
            muon_from_tau.append(2)
        else:
            muon_from_tau.append(0)  # 0 for none
            intersection = [0, 0, 0]
        intersections.append(intersection)
        # now find angle between tauMu and the plane
        normal_to_plane = np.cross(vector_plane_lb, momentum_pkmu)
        angle_mu_plane_normal = np.arccos(np.dot(normal_to_plane, momentum_tauMu) / (
                np.linalg.norm(normal_to_plane) * np.linalg.norm(momentum_tauMu)))
        angle_mu_plane_normal = np.degrees(angle_mu_plane_normal)
        angle_mu_plane = 90 - angle_mu_plane_normal if angle_mu_plane_normal < 90 else angle_mu_plane_normal - 90
        angles_mu.append(angle_mu_plane)
        calc.append(possible_calculation_u)
        dets.append(determinant)

    data_frame['tau_decay_point'] = intersections
    data_frame['angle_to_plane'] = angles_mu
    data_frame['calc'] = calc
    data_frame['dets'] = dets
    data_frame = data_frame.reset_index(drop=True)
    # np.save('angle_cleaned_data.npy', data_frame['angle_to_plane'].values)
    # plt.hist(angles_mu, bins=100, range=[0, 4])
    # plt.xlabel('Angle between tauMu and the plane')
    # plt.ylabel('Occurrences')
    # plt.show()
    return data_frame


def transverse_momentum(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the transverse momentum of the potential tau
    Finds the pkmu transverse momentum (with respect to the Lb line of flight)
    This transverse momentum is the opposite of the tau transverse momentum
    :param data_frame:
    :return:
    """
    pkmu = [np.sqrt(data_frame['pKmu_P'] ** 2), data_frame['pKmu_PX'], data_frame['pKmu_PY'], data_frame['pKmu_PZ']]
    vectors = data_frame['vectors']
    transverse_momenta = []
    pkmu_momentum = momentum(pkmu)
    for i in range(len(data_frame)):
        par_vector = vectors[i] / np.linalg.norm(vectors[i])  # unit vector of the parallel direction vector
        pkmu_vector = np.array(pkmu_momentum.loc[i])
        transverse_mom = pkmu_vector - np.dot(pkmu_vector, par_vector) * par_vector
        transverse_momenta.append(-transverse_mom)
    data_frame['transverse_momentum'] = transverse_momenta
    return data_frame


def tau_momentum_mass(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the total momentum of the tau
    Uses the transverse momentum and the angle between the tau direction and the Lb line of flight
    :param data_frame:
    :return:
    """
    angles, tau_p, tau_distances_travelled = [], [], []
    tau_p_x, tau_p_y, tau_p_z = [], [], []
    for i in range(len(data_frame)):
        temp_series = data_frame.loc[i]
        end_xyz = [temp_series['pKmu_ENDVERTEX_X'], temp_series['pKmu_ENDVERTEX_Y'], temp_series['pKmu_ENDVERTEX_Z']]
        tau_vector = temp_series['tau_decay_point'] - end_xyz
        vector = temp_series['vectors']
        tau_distance = np.linalg.norm(tau_vector)
        tau_distances_travelled.append(tau_distance * np.sign(np.dot(tau_vector, vector)))
        # tau_distances_travelled.append(tau_distance)
        angle = np.arccos(np.dot(tau_vector, vector) / (np.linalg.norm(tau_vector) * np.linalg.norm(vector)))
        angles.append(angle)
        unit_l = vector / np.linalg.norm(vector)
        p_transverse = np.linalg.norm(temp_series['transverse_momentum'])
        tau_mom = p_transverse / np.tan(angle) * unit_l + temp_series['transverse_momentum']
        tau_p_x.append(tau_mom[0])
        tau_p_y.append(tau_mom[1])
        tau_p_z.append(tau_mom[2])
        tau_p.append(np.linalg.norm(tau_mom))

    data_frame['tau_PX'], data_frame['tau_PY'], data_frame['tau_PZ'] = tau_p_x, tau_p_y, tau_p_z
    data_frame['tau_P'] = tau_p

    # plt.hist(data_frame['tau_P'], bins=200, range=[0, 200000], density=True)
    # plt.title('Momenta of the the "taus"')
    # plt.show()
    data_frame['tau_distances_travelled'] = tau_distances_travelled
    data_frame['angles'] = angles

    # np.save('distance_taus_super_cleaned_data.npy', data_frame['tau_distances_travelled'].values)
    # np.save('distance_taus_pkmu_below_2300.npy', data_frame['tau_distances_travelled'].values)
    # np.save('distance_real_taus.npy', data_frame['tau_distances_travelled'].values)
    # np.save('distance_jpsi_taus.npy', data_frame['tau_distances_travelled'].values)

    # plt.hist2d(data_frame['tau_distances_travelled'], data_frame['lb_distances'], bins=20, range=[[0, 30], [0, 30]])
    # plt.title('Distance travelled by the "taus" versus lb')
    # plt.show()
    data_frame['vector_muTau'] = data_frame[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    data_frame['tauMu_reference_point'] = data_frame[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    data_frame['pkmu_endvertex_point'] = data_frame[
        ['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    data_frame['pkmu_direction'] = data_frame[['pKmu_PX', 'pKmu_PY', 'pKmu_PZ']].values.tolist()
    data_frame['impact_parameter_thingy'] = line_point_distance(vector=data_frame['vector_muTau'],
                                                                vector_point=data_frame['tauMu_reference_point'],
                                                                point=data_frame['pkmu_endvertex_point'],
                                                                direction=data_frame['pkmu_direction'])
    # plt.hist(data_frame['impact_parameter_thingy'], range=[-0.4, 0.4], bins=100)
    # plt.ylabel('occurrences')
    # plt.xlabel('IP*')
    # plt.show()
    # np.save('ipstar_250_cleaned_data_nolb.npy', data_frame['impact_parameter_thingy'].values)

    # plt.hist2d(data_frame['impact_parameter_thingy'], data_frame['tau_distances_travelled'],
    #            range=[[-0.1, 0.1], [-7.5, 7.5]], bins=50, norm=LogNorm())
    # plt.show()
    data_frame, ip_cols = return_all_ip(data_frame)
    # for col in ip_cols:
    #     df_to_plot = data_frame[np.sign(data_frame['proton_ID']) == np.sign(data_frame['mu1_ID'])]
    #     plt.hist(np.load(col + 'Lb_MC_data.npy'), bins=100, range=[-0.4, 0.4], density=True, alpha=0.3,
    #              label='pkmumu MC data')
    #     plt.hist(df_to_plot[col], range=[-0.4, 0.4], bins=50, label='cleaned data', density=True, histtype='step')
    #     plt.ylabel('occurrences')
    #     plt.xlabel(col + ' p and mu1 same charge')
    #     plt.legend()
    #     # plt.savefig(col + 'samesign.png')
    #     plt.show()
    # np.save(col + 'background_cleaned_data.npy', data_frame[col].values)

    # real_taus, fake_taus = np.load('distance_real_taus_plus_minus.npy'), np.load('distance_fake_taus_plus_minus.npy')
    # _bins, _range = 200, [-10, 10]
    # plt.hist(fake_taus, bins=_bins, range=_range, density=True, label='pkmumu MC data', alpha=0.3)
    # plt.hist(real_taus, bins=_bins, range=_range, density=True, label='B MC data', histtype='step')
    # plt.hist(data_frame['tau_distances_travelled'], bins=20, range=_range, density=True, label='cleaned data',
    #          histtype='step')
    # plt.legend()
    # plt.xlabel('Tau FD')
    # plt.show()
    return data_frame


def momentum(frame_array: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Returns momentum of data frame made of [energy, Px Py, Pz]
    :param frame_array:
    :return:
    """
    mom = pd.concat([frame_array[1], frame_array[2], frame_array[3]], axis=1)
    mom.columns = ['X', 'Y', 'Z']
    return mom


def plot_b_result(data_frame: pd.DataFrame):
    """
    Plots mass results for the B particles
    :param data_frame:
    :return:
    """
    # print(len(data_frame))
    # data_frame['mutau_mass'] = get_mass(data_frame, [['mu1_P', 'mu'], ['tau_P', 'tau']])
    # data_frame = data_frame[data_frame['mutau_mass'] < masses['B'] - masses['pi'] - masses['K']]
    # print(len(data_frame))
    # data_frame['pitau_mass'] = get_mass(data_frame, [['proton_P', 'pi'], ['tau_P', 'tau']])
    # data_frame = data_frame[data_frame['pitau_mass'] < masses['B'] - masses['mu'] - masses['K']]
    # print(len(data_frame))
    # data_frame['Ktau_mass'] = get_mass(data_frame, [['Kminus_P', 'K'], ['tau_P', 'tau']])
    # data_frame = data_frame[data_frame['Ktau_mass'] < masses['B'] - masses['pi'] - masses['mu']]
    # print(len(data_frame))
    # df_for_bdt = data_frame[
    #     ['tau_distances_travelled', 'impact_parameter_thingy', 'pKmu_ENDVERTEX_CHI2', 'Lb_pmu_ISOLATION_BDT1',
    #      'proton_IPCHI2_OWNPV', 'Kminus_IPCHI2_OWNPV', 'mu1_IPCHI2_OWNPV', 'tauMu_IPCHI2_OWNPV', 'pKmu_IPCHI2_OWNPV',
    #      'Lb_FDCHI2_OWNPV']]
    # bdt = obtain_bdt()
    # predictions = bdt.decision_function(df_for_bdt)
    # predictions = bdt.predict_proba(df_for_bdt)
    # data_frame['predictions_from_bdt'] = predictions
    # data_frame = data_frame[data_frame['predictions_from_bdt'] > 0.62]
    print('final length of data', len(data_frame))
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    data_frame['b_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame = data_frame.reset_index(drop=True)
    data_frame['B_low'] = 'middle'
    data_frame.loc[data_frame['b_mass'] < 8000, 'B_low'] = 'low'
    data_frame.loc[data_frame['b_mass'] > 28000, 'B_low'] = 'high'
    tau_decay_points = data_frame['tau_decay_point'].values
    pkmu_endvertex_point = data_frame[['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    data_frame['tvx'] = [(tau_decay_points[i] - pkmu_endvertex_point[i])[0] for i in range(len(tau_decay_points))]
    data_frame['tvy'] = [(tau_decay_points[i] - pkmu_endvertex_point[i])[1] for i in range(len(tau_decay_points))]
    data_frame['tvz'] = [(tau_decay_points[i] - pkmu_endvertex_point[i])[2] for i in range(len(tau_decay_points))]
    data_frame['lvx'] = [(data_frame.loc[i, 'vectors'])[0] for i in range(len(tau_decay_points))]
    data_frame['lvy'] = [(data_frame.loc[i, 'vectors'])[1] for i in range(len(tau_decay_points))]
    data_frame['lvz'] = [(data_frame.loc[i, 'vectors'])[2] for i in range(len(tau_decay_points))]
    print(data_frame.groupby(['B_low']).std().loc[:, ['tvx', 'tvy', 'tvz', 'calc']])
    print(data_frame.groupby(['B_low']).mean().loc[:, ['tvx', 'tvy', 'tvz', 'calc']])
    print(data_frame.groupby(['B_low']).median().loc[:, ['tvx', 'tvy', 'tvz', 'calc']])
    print(data_frame.groupby(['B_low']).max().loc[:, ['tvx', 'tvy', 'tvz', 'calc']])
    print(data_frame.groupby(['B_low']).min().loc[:, ['tvx', 'tvy', 'tvz', 'calc']])
    # data_frame = data_frame[data_frame['tvz'] > 0]
    # data_frame = data_frame[data_frame['tau_distances_travelled'] > 0]
    data_frame = geometrical_cuts(data_frame)
    print(f'Final length : {len(data_frame)}')

    sns.pairplot(data=data_frame,
                 x_vars=['tau_distances_travelled', 'angle_to_plane', 'tvz', 'calc', 'angles'],
                 y_vars=['tau_distances_travelled', 'angle_to_plane', 'tvz', 'calc', 'angles'],
                 hue='B_low')
    sns.pairplot(data=data_frame, x_vars=['tvx', 'tvy', 'tvz'], y_vars=['tvx', 'tvy', 'tvz'], hue='B_low')
    sns.pairplot(data=data_frame, x_vars=['lvx', 'lvy', 'lvz'], y_vars=['lvx', 'lvy', 'lvz'], hue='B_low')
    sns.pairplot(data=data_frame, x_vars=['tauMu_PX', 'tauMu_PY', 'tauMu_PZ'],
                 y_vars=['tauMu_PX', 'tauMu_PY', 'tauMu_PZ'], hue='B_low')
    sns.pairplot(data=data_frame, x_vars=['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ'],
                 y_vars=['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ'], hue='B_low')
    plt.show()

    print(data_frame['tau_distances_travelled'].describe())
    # data_frame = data_frame[data_frame['tau_distances_travelled'] > 0]
    print(data_frame['angle_to_plane'].describe())
    print(data_frame['lb_distances'].describe())
    # pd.to_pickle(data_frame, 'cleaned_B_df_kmu_removed_stretched.gz')

    # np.save('B_MC_mass.npy', data_frame['b_mass'].values)
    # np.save('B_MC_mass_taurestriction_nokmu.npy', data_frame['b_mass'].values)
    # np.save('B_MC_mass_taurestriction.npy', data_frame['b_mass'].values)
    # np.save('B_MC_mass_all_cleaning.npy', data_frame['b_mass'].values)
    print('final length of the data', len(data_frame['b_mass']))
    # np.save('b_mass_kmu_cut.npy', data_frame['b_mass'].values)
    n, b, p = plt.hist(data_frame['b_mass'], bins=100, range=[4000, 15000])
    print(np.quantile(data_frame['b_mass'], 0.30) - masses['B'] + masses['Lb'])
    print(np.quantile(data_frame['b_mass'], 0.70) - masses['B'] + masses['Lb'])
    print(np.quantile(data_frame['b_mass'], (1 - 0.954499736103642) / 2) - masses['B'] + masses['Lb'])
    print(np.quantile(data_frame['b_mass'], 1 - (1 - 0.954499736103642) / 2) - masses['B'] + masses['Lb'])
    plt.vlines(5279, ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{B}$')
    plt.ylabel('occurrences')
    plt.show()
    data_frame['kpi_mass'] = get_mass(data_frame=data_frame,
                                      particles_associations=[['Kminus_P', 'K'], ['proton_P', 'pi']])
    print(data_frame['kpi_mass'])
    n, b, p = plt.hist(data_frame['kpi_mass'], bins=100)
    plt.vlines(892, ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{k\\pi}$')
    plt.ylabel('occurrences')
    plt.show()

    # # csv_storage = data_frame[['tau_distances_travelled', 'impact_parameter_thingy']]
    # csv_storage = data_frame[
    #     ['tau_distances_travelled', 'impact_parameter_thingy', 'pKmu_ENDVERTEX_CHI2', 'Lb_pmu_ISOLATION_BDT1',
    #      'proton_IPCHI2_OWNPV', 'Kminus_IPCHI2_OWNPV', 'mu1_IPCHI2_OWNPV', 'tauMu_IPCHI2_OWNPV', 'pKmu_IPCHI2_OWNPV',
    #      'Lb_FDCHI2_OWNPV']]
    # csv_storage.to_csv('B_MC_mass_taurestriction_nokmu.csv')


def geometrical_cuts(data_frame):
    data_frame = data_frame[data_frame['angles'] < 0.5 * np.pi]
    tau_decay_points = data_frame['tau_decay_point'].values
    pkmu_endvertex_point = data_frame[['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    data_frame['tvx'] = [(tau_decay_points[i] - pkmu_endvertex_point[i])[0] for i in range(len(tau_decay_points))]
    data_frame['tvy'] = [(tau_decay_points[i] - pkmu_endvertex_point[i])[1] for i in range(len(tau_decay_points))]
    data_frame['tvz'] = [(tau_decay_points[i] - pkmu_endvertex_point[i])[2] for i in range(len(tau_decay_points))]
    data_frame = additional_pkmu_cleaning(data_frame)
    return data_frame


def plot_result(data_frame: pd.DataFrame):
    """
    Plots mass results for the Lb particles
    :param data_frame:
    :return:
    """
    # data_frame['mutau_mass'] = get_mass(data_frame, [['mu1_P', 'mu'], ['tau_P', 'tau']])
    # data_frame = data_frame[data_frame['mutau_mass'] < masses['Lb'] - masses['proton'] - masses['K']]
    # data_frame['ptau_mass'] = get_mass(data_frame, [['proton_P', 'proton'], ['tau_P', 'tau']])
    # data_frame = data_frame[data_frame['ptau_mass'] < masses['Lb'] - masses['mu'] - masses['K']]
    # data_frame['Ktau_mass'] = get_mass(data_frame, [['Kminus_P', 'K'], ['tau_P', 'tau']])
    # data_frame = data_frame[data_frame['Ktau_mass'] < masses['Lb'] - masses['proton'] - masses['mu']]
    # # df_for_bdt = data_frame[['tau_distances_travelled', 'impact_parameter_thingy']]
    data_frame = geometrical_cuts(data_frame)
    df_for_bdt = data_frame[
        ['tau_distances_travelled', 'impact_parameter_thingy', 'pKmu_ENDVERTEX_CHI2', 'Lb_pmu_ISOLATION_BDT1',
         'proton_IPCHI2_OWNPV', 'Kminus_IPCHI2_OWNPV', 'mu1_IPCHI2_OWNPV', 'tauMu_IPCHI2_OWNPV', 'pKmu_IPCHI2_OWNPV',
         'Lb_FDCHI2_OWNPV']]

    xgb = xgboost.XGBClassifier()

    xgb.load_model('bdt\\classifier_0.pkl')
    model_y_pred = xgb.predict_proba(df_for_bdt)[:, 1]
    model_y_pred_class = model_y_pred > 0.915
    data_frame = data_frame.loc[model_y_pred_class]
    # bdt1 = obtain_bdt(1)
    # bdt2 = obtain_bdt(2)
    # predictions1 = bdt1.predict_proba(df_for_bdt)
    # predictions2 = bdt2.predict_proba(df_for_bdt)
    # data_frame['predictions_from_bdt1'] = predictions1[:, 1]
    # data_frame['predictions_from_bdt2'] = predictions2[:, 1]
    # print(data_frame['predictions_from_bdt1'].describe())
    # print(data_frame['predictions_from_bdt2'].describe())
    # data_frame = data_frame[data_frame['predictions_from_bdt1'] > 0.65]
    # # data_frame = data_frame[data_frame['predictions_from_bdt2'] > 0.65]
    # data_frame = data_frame.reset_index(drop=True)
    # # combs = check_for_both_charges(data_frame)
    # # data_frame = data_frame.drop(np.array(combs.flatten()))
    # # data_frame = data_frame[data_frame['tau_distances_travelled'] > 0]
    # data_frame = data_frame[data_frame['tau_distances_travelled'] > 0]
    #

    data_frame['pkmu'] = get_mass(data_frame,
                                  particles_associations=[['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']])
    data_frame['pktauMu'] = get_mass(data_frame,
                                     particles_associations=[['Kminus_P', 'K'], ['proton_P', 'proton'],
                                                             ['tauMu_P', 'mu']])
    df_mu = data_frame[np.sign(data_frame['proton_ID']) != np.sign(data_frame['mu1_ID'])]
    df_tauMu = data_frame[np.sign(data_frame['proton_ID']) != np.sign(data_frame['tauMu_ID'])]
    # plt.hist([df_mu['pkmu'], df_tauMu['pktauMu']], bins=100, stacked=True, color=['C0', 'C0'], range=[2700, 5000])
    # plt.hist([df_mu[(df_mu['pkmumu_mass'] > 5620 - 40) & (data_frame['pkmumu_mass'] < 5620 + 40)]['pkmu'],
    #           df_tauMu[(df_tauMu['pkmumu_mass'] > 5620 - 40) & (df_tauMu['pkmumu_mass'] < 5620 + 40)][
    #               'pktauMu']], bins=100, stacked=True, color=['C1', 'C1'], range=[2700, 5000])
    # plt.xlabel('pKmu mass')
    # plt.xlim(right=5000)
    # plt.show()

    data_frame['tauMu_mass'] = get_mass(data_frame=data_frame,
                                        particles_associations=[['mu1_P', 'mu'], ['tau_P', 'tau']])
    plt.hist(data_frame['tauMu_mass'], range=[1500, 7000], bins=50)
    plt.xlabel('$m_{\\mu\\tau}$')
    plt.show()

    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pkmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    length_pkmumu_signal = len(
        data_frame[(data_frame['pkmumu_mass'] > 5620 - 40) & (data_frame['pkmumu_mass'] < 5620 + 40)])
    try:
        print(len(data_frame), length_pkmumu_signal / len(data_frame))
    except ZeroDivisionError:
        print('nothing seems to be outside of the pkmumu peak')
    minimum_mass_pkmumu = masses['proton'] + masses['K'] + masses['mu'] + masses['mu']
    plt.figure(figsize=(3.5, 3), dpi=300)
    n, b, p = plt.hist(data_frame['pkmumu_mass'], bins=100, range=[minimum_mass_pkmumu - 100, 9000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n), linewidth=0.5)
    plt.xlabel('$m_{pK\\mu\\mu}$')
    plt.ylabel('occurrences')
    plt.tight_layout()
    plt.show()

    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    data_frame['pkmutau_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    print('final length of the data', len(data_frame))
    # print(data_frame['pkmutau_mass'].describe())
    minimum_mass_pkmutau = masses['proton'] + masses['K'] + masses['mu'] + masses['tau']
    plt.figure(figsize=(3.5, 3), dpi=300)
    plt.hist(
        data_frame[(data_frame['pkmumu_mass'] < 5620 - 40) | (data_frame['pkmumu_mass'] > 5620 + 40)]['pkmutau_mass'],
        bins=100, range=[minimum_mass_pkmutau - 100, 30000])
    # n, b, p = plt.hist(data_frame['pkmutau_mass'], bins=100, range=[minimum_mass_pkmutau - 100, 15000])
    plt.axvline(masses['Lb'], c='k')
    plt.xlabel('$m_{pK\\mu\\tau}$')
    # np.save('pkmutau_cleaned_data_nokmucleaning.npy', (
    # data_frame[(data_frame['pkmumu_mass'] < 5620 - 40) | (data_frame['pkmumu_mass'] > 5620 + 40)][
    #     'pkmutau_mass']).values)
    # plt.gca().axvspan(5736, 14138, color='k', alpha=0.3)
    # plt.gca().axvspan(5705, 9896, color='k', alpha=0.3)
    plt.gca().axvspan(5111, 7230, color='k', alpha=0.3)
    plt.ylabel('occurrences')
    plt.tight_layout()
    plt.show()

    # pd.to_pickle(data_frame, 'cleaned_Lb_df_removed_kmu.gz')

    # background = data_frame[(data_frame['pkmutau_mass'] > 9896) | (data_frame['pkmutau_mass'] < 5705)]
    # background = background[(background['pkmumu_mass'] < 5620 - 40) | (background['pkmumu_mass'] > 5620 + 40)]
    # n, b, p = plt.hist(background['pkmutau_mass'], bins=100, density=True)
    # print(len(background))
    # print(np.unique(n, return_counts=True))
    # (_lambda, e), _ = spo.curve_fit(exponential, ((b[1:] + b[:-1]) / 2), n, p0=[1 / 600, 7000])
    # print(_lambda, e)
    # x = np.linspace(np.min(background['pkmutau_mass']), np.max(background['pkmutau_mass']), 150)
    # plt.plot(x, exponential(x, _lambda, e))
    # plt.show()
    # plt.xlim(right=15000)

    # # csv_storage = data_frame[data_frame['pkmutau_mass'] > masses['Lb']+3*peak_supposed_width]
    # # csv_storage = data_frame[data_frame['pkmutau_mass'] > 7000]
    # csv_storage = data_frame[(data_frame['pkmutau_mass'] > 9896) | (data_frame['pkmutau_mass'] < 5705)]
    # csv_storage = data_frame[(data_frame['pkmutau_mass'] > 7230) | (data_frame['pkmutau_mass'] < 5111)]
    # # csv_storage = csv_storage[['tau_distances_travelled', 'impact_parameter_thingy']]
    # csv_storage = csv_storage[
    #     ['tau_distances_travelled', 'impact_parameter_thingy', 'pKmu_ENDVERTEX_CHI2', 'Lb_pmu_ISOLATION_BDT1',
    #      'proton_IPCHI2_OWNPV', 'Kminus_IPCHI2_OWNPV', 'mu1_IPCHI2_OWNPV', 'tauMu_IPCHI2_OWNPV', 'pKmu_IPCHI2_OWNPV',
    #      'Lb_FDCHI2_OWNPV']]
    # csv_storage.to_csv('cleaned_data_2sigma_nokmu.csv')

    df_no_lb = data_frame[(data_frame['pkmumu_mass'] < 5620 - 40) | (data_frame['pkmumu_mass'] > 5620 + 40)]
    plt.hist(data_frame['dimuon_mass'], bins=100)
    plt.xlabel('dimuon mass')
    plt.show()
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    df_no_lb['pikmumu_mass'] = get_mass(df_no_lb, particles_associations)
    plt.hist(df_no_lb['pikmumu_mass'], bins=100)
    plt.xlabel('$m_{K\\pi\\mu\\mu}$')
    plt.show()
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    plt.hist(get_mass(df_no_lb, particles_associations), bins=100)
    plt.xlabel('$m_{p\\pi\\mu\\mu}$')
    plt.show()
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'pi'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    plt.hist(get_mass(df_no_lb, particles_associations), bins=100)
    plt.xlabel('$m_{\\pi\\pi\\mu\\mu}$')
    plt.show()
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    df_no_lb['kkmumu_mass'] = get_mass(df_no_lb, particles_associations)
    plt.hist(df_no_lb['kkmumu_mass'], bins=100)
    plt.xlabel('$m_{KK\\mu\\mu}$')
    plt.show()
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'K']]
    df_no_lb['kk_mass'] = get_mass(df_no_lb, particles_associations)
    plt.hist(df_no_lb['kk_mass'], bins=100)
    plt.xlabel('$m_{KK}$')
    plt.show()
    df_no_lb['pik_mass'] = get_mass(df_no_lb, [['Kminus_P', 'K'], ['proton_P', 'pi']])
    plt.hist(df_no_lb['pik_mass'], bins=100)
    plt.xlabel('$m_{K\\pi}$')
    plt.show()
    df_no_lb['pikmu_mass'] = get_mass(df_no_lb, [['Kminus_P', 'K'], ['proton_P', 'pi'], ['mu1_P', 'mu']])
    plt.hist(df_no_lb[np.sign(df_no_lb['proton_ID']) == np.sign(df_no_lb['mu1_ID'])]['pikmu_mass'], bins=100)
    plt.xlabel('$m_{K\\pi\\mu}$')
    plt.show()
    plt.hist(df_no_lb[np.sign(df_no_lb['Kminus_ID']) == np.sign(df_no_lb['mu1_ID'])]['pikmu_mass'], bins=100)
    plt.xlabel('$m_{K\\pi\\mu}$')
    plt.show()
    plt.hist2d(df_no_lb['kk_mass'], df_no_lb['kkmumu_mass'], bins=40)
    plt.xlabel('$m_{KK}$')
    plt.ylabel('$m_{KK\\mu\\mu}$')
    plt.show()
    plt.hist2d(df_no_lb['pik_mass'], df_no_lb['pikmumu_mass'], bins=40)
    plt.xlabel('$m_{K\\pi}$')
    plt.ylabel('$m_{K\\pi\\mu\\mu}$')
    plt.show()

    p_and_mu1_same_charge = data_frame[np.sign(data_frame['proton_ID']) != np.sign(data_frame['mu1_ID'])]
    p_and_mu1_diff_charge = data_frame[np.sign(data_frame['proton_ID']) == np.sign(data_frame['mu1_ID'])]
    n, b, p = plt.hist(p_and_mu1_same_charge['pkmutau_mass'], bins=100, range=[minimum_mass_pkmutau - 100, 15000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n), linewidth=0.5)
    plt.xlabel('$m_{pK\\mu\\tau}$ where p and mu have the same charge')
    plt.show()
    n, b, p = plt.hist(p_and_mu1_diff_charge['pkmutau_mass'], bins=100, range=[minimum_mass_pkmutau - 100, 15000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n), linewidth=0.5)
    plt.xlabel('$m_{pK\\mu\\tau}$ where p and mu have opposite charges')
    plt.show()
    n, b, p = plt.hist(p_and_mu1_same_charge['pkmumu_mass'], bins=100, range=[minimum_mass_pkmutau - 100, 15000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n), linewidth=0.5)
    plt.xlabel('$m_{pK\\mu\\mu}$ where p and mu1 have the same charge')
    plt.show()
    n, b, p = plt.hist(p_and_mu1_diff_charge['pkmumu_mass'], bins=100, range=[minimum_mass_pkmutau - 100, 15000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n), linewidth=0.5)
    plt.xlabel('$m_{pK\\mu\\mu}$ where p and mu1 have opposite charges')
    plt.show()
    plt.hist2d(data_frame['pkmumu_mass'], data_frame['pkmutau_mass'], bins=20, range=[[2000, 8000], [4000, 8000]],
               norm=LogNorm())
    plt.axvline(masses['Lb'])
    plt.axhline(masses['Lb'])
    plt.xlabel('$m_{pK\\mu\\mu}$')
    plt.ylabel('$m_{pK\\mu\\tau}$')
    plt.colorbar()
    plt.show()

    new_frame = data_frame[(data_frame['pkmutau_mass'] < 6500) & (data_frame['pkmutau_mass'] > 4500)]
    new_frame.hist(bins=50, grid=False, column=['pKmu_IPCHI2_OWNPV', 'pKmu_ENDVERTEX_CHI2'])
    plt.show()
    return


def additional_pkmu_cleaning(data_frame):
    _tau_dir_vars = ['_tau_dir_x', '_tau_dir_y', '_tau_dir_z']
    _tau_direction = data_frame.loc[:, ['tvx', 'tvy', 'tvz']]
    data_frame[_tau_dir_vars] = _tau_direction.divide((_tau_direction ** 2).sum(axis=1) ** 0.5, axis='rows')
    _tau_direction = data_frame.loc[:, _tau_dir_vars]
    _pKmu_dir_vars = ['_pKmu_dir_x', '_pKmu_dir_y', '_pKmu_dir_z']
    _pKmu_direction = data_frame.loc[:, ['pKmu_PX', 'pKmu_PY', 'pKmu_PZ']]
    data_frame[_pKmu_dir_vars] = _pKmu_direction.divide((_pKmu_direction ** 2).sum(axis=1) ** 0.5, axis='rows')
    _pKmu_direction = data_frame.loc[:, _pKmu_dir_vars]
    _vectors_array_dtype = data_frame.loc[:, 'vectors']
    _vectors_array = np.vstack(_vectors_array_dtype.to_numpy())
    data_frame.loc[:, '_lb_dir_x'] = _vectors_array[:, 0]
    data_frame.loc[:, '_lb_dir_y'] = _vectors_array[:, 1]
    data_frame.loc[:, '_lb_dir_z'] = _vectors_array[:, 2]
    _lb_dir_vars = ['_lb_dir_x', '_lb_dir_y', '_lb_dir_z']
    _lb_direction = data_frame.loc[:, ['_lb_dir_x', '_lb_dir_y', '_lb_dir_z']]
    data_frame[_lb_dir_vars] = _lb_direction.divide((_lb_direction ** 2).sum(axis=1) ** 0.5, axis='rows')
    _lb_direction = data_frame.loc[:, _lb_dir_vars]

    __tau_dir = data_frame.loc[:, _tau_dir_vars]
    __pKmu_dir = data_frame.loc[:, _pKmu_dir_vars]
    __lb_dir = data_frame.loc[:, _lb_dir_vars]

    __lb_dir_array = __lb_dir.to_numpy()
    __pKmu_dir_array = __pKmu_dir.to_numpy()

    _pKmu_longitudinal = __lb_dir_array * (__pKmu_dir_array * __lb_dir_array).sum(axis=1).reshape((-1, 1))
    ideal_opposite_dir = -(__pKmu_dir_array - _pKmu_longitudinal)
    alignment = (ideal_opposite_dir * __tau_dir.to_numpy()).sum(axis=1)
    alignment = np.sign(alignment)
    print(np.unique(alignment, return_counts=True))
    data_frame['alignment'] = alignment
    data_frame = data_frame[data_frame['alignment'] == 1]
    return data_frame
