import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from background_reduction import reduce_background, b_cleaning
from data_loader import load_data, add_branches
from masses import masses, get_mass

from error_propagation import error_propagation_intersection, error_propagation_transverse_momentum, \
    error_propagation_momentum_mass
from matrix_calculations import find_determinant_of_dir_matrix, find_inverse_of_dir_matrix


def retrieve_vertices(data_frame):
    plt.hist(data_frame['Lb_FD_OWNPV'], bins=30, range=[0, 60])
    plt.title('Lb_FD_OWNPV')
    plt.show()
    print(data_frame['Lb_FD_OWNPV'].describe())
    print(np.percentile(data_frame['Lb_FD_OWNPV'], 90))
    plt.hist(data_frame['tauMu_P'], bins=200, range=[0, 200000], density=True)
    plt.title('Momenta of the the "taus"')
    plt.show()
    all_distances, vectors, errors = [], [], []
    print('initial length of data frame: ', len(data_frame))
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        pv_xyz = [ts['Lb_OWNPV_X'], ts['Lb_OWNPV_Y'], ts['Lb_OWNPV_Z']]
        end_xyz = [ts['pKmu_ENDVERTEX_X'], ts['pKmu_ENDVERTEX_Y'], ts['pKmu_ENDVERTEX_Z']]
        # end_xyz = [ts['Lb_ENDVERTEX_X'], ts['Lb_ENDVERTEX_Y'], ts['Lb_ENDVERTEX_Z']]
        # lb_endvertex = [ts['Lb_ENDVERTEX_X'], ts['Lb_ENDVERTEX_Y'], ts['Lb_ENDVERTEX_Z']]
        # print(end_xyz, lb_endvertex)
        # print(1-(np.array(lb_endvertex)/np.array(end_xyz)))
        # errors_pv = [ts['Lb_OWNPV_XERR'], ts['Lb_OWNPV_YERR'], ts['Lb_OWNPV_ZERR']]
        # errors_end = [ts['pKmu_REFP_COVXX'], ts['pKmu_REFP_COVYY'], ts['pKmu_REFP_COVZZ']]
        distance = np.linalg.norm(np.array(pv_xyz) - np.array(end_xyz))
        vector = np.array(end_xyz) - np.array(pv_xyz)
        # error_x = np.sqrt((errors_pv[0]) ** 2 + (errors_end[0]))
        # error_y = np.sqrt((errors_pv[1]) ** 2 + (errors_end[1]))
        # error_z = np.sqrt((errors_pv[2]) ** 2 + (errors_end[2]))
        # errors.append([error_x, error_y, error_z])
        vectors.append(vector)
        all_distances.append(distance)
    data_frame['lb_distances'] = all_distances
    data_frame['vectors'] = vectors
    # data_frame['vectors_errors'] = errors
    # data_frame = data_frame.drop('lb_distances', axis=1)
    # data_frame = data_frame[data_frame['lb_distances'] < 5]
    # data_frame = data_frame[data_frame['lb_distances'] > 5]
    # data_frame = data_frame[data_frame['lb_distances'] < 25]
    # data_frame = data_frame[data_frame['lb_distances'] < 40]
    # data_frame = data_frame[data_frame['lb_distances'] > 25]
    # data_frame = data_frame[data_frame['lb_distances'] > 40]
    # n, b, t = plt.hist(data_frame['Lb_FD_OWNPV'], bins=100, range=[0, 100])
    # plt.title('Lb flight distance for the jpsi data')
    # plt.ylabel('occurrences')
    # plt.show()
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def line_plane_intersection(data_frame):
    intersections, muon_from_tau = [], []
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        end_xyz = [ts['pKmu_ENDVERTEX_X'], ts['pKmu_ENDVERTEX_Y'], ts['pKmu_ENDVERTEX_Z']]
        # end_xyz = [ts['Lb_ENDVERTEX_X'], ts['Lb_ENDVERTEX_Y'], ts['Lb_ENDVERTEX_Z']]
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
            # possible_intersection = np.linalg.solve(coefficient_matrix, ordinate)
            # possible_calculation_s = sum([inverse_matrix[0][j] * ordinate[j] for j in range(3)])
            # possible_calculation_t = sum([inverse_matrix[1][j] * ordinate[j] for j in range(3)])
            possible_calculation_u = sum([inverse_matrix[2][j] * ordinate[j] for j in range(3)])
            possible_intersection = np.array(momentum_tauMu) * possible_calculation_u + np.array(point_tau_mu)
            # sigma_i = error_propagation_intersection(data_frame=data_frame, inverse_A=inverse_matrix*determinant,
            #                                          determinant=determinant, ordinate=ordinate, i=i,
            #                                          possible_calculation_u=possible_calculation_u)
        else:
            possible_intersection = False
        if possible_intersection is not False:
            intersection = possible_intersection
            muon_from_tau.append(2)
        else:
            muon_from_tau.append(0)  # 0 for none
            intersection = [0, 0, 0]
        intersections.append(intersection)

    data_frame['muon_from_tau'] = muon_from_tau
    data_frame['tau_decay_point'] = intersections
    data_frame = data_frame[data_frame['muon_from_tau'] > 0]
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def transverse_momentum(data_frame):
    pkmu = [np.sqrt(data_frame['pKmu_P'] ** 2), data_frame['pKmu_PX'], data_frame['pKmu_PY'], data_frame['pKmu_PZ']]
    vectors = data_frame['vectors']
    transverse_momenta = []
    pkmu_momentum = momentum(pkmu)
    for i in range(len(data_frame)):
        par_vector = vectors[i] / np.linalg.norm(vectors[i])
        pkmu_vector = np.array(pkmu_momentum.loc[i])
        par = np.dot(pkmu_vector, par_vector) * par_vector
        transverse_mom = pkmu_vector - par
        # t_mom_err = error_propagation_transverse_momentum(data_frame=data_frame, vectors=vectors, i=i,
        #                                                   pkmu_vector=pkmu_vector)
        transverse_momenta.append(-transverse_mom)
    data_frame['transverse_momentum'] = transverse_momenta

    return data_frame


def tau_momentum_mass(data_frame):
    angles, tau_p, tau_distances_travelled = [], [], []
    tau_p_x, tau_p_y, tau_p_z = [], [], []
    for i in range(len(data_frame)):
        temp_series = data_frame.loc[i]
        end_xyz = [temp_series['pKmu_ENDVERTEX_X'], temp_series['pKmu_ENDVERTEX_Y'], temp_series['pKmu_ENDVERTEX_Z']]
        # end_xyz = [temp_series['Lb_ENDVERTEX_X'], temp_series['Lb_ENDVERTEX_Y'], temp_series['Lb_ENDVERTEX_Z']]
        tau_vector = temp_series['tau_decay_point'] - end_xyz
        vector = temp_series['vectors']
        tau_distance = np.linalg.norm(tau_vector)
        # momentum_tauMu = [temp_series['tauMu_PX'], temp_series['tauMu_PY'], temp_series['tauMu_PZ']]
        # vector_plane_lb = temp_series['vectors']
        # vector_with_mu1 = [temp_series['pKmu_PX'], temp_series['pKmu_PY'], temp_series['pKmu_PZ']]
        # point_tau_mu = [temp_series['tauMu_REFPX'], temp_series['tauMu_REFPY'], temp_series['tauMu_REFPZ']]
        # det = find_determinant_of_dir_matrix(vector_plane_lb=vector_plane_lb, momentum_pkmu=vector_with_mu1,
        #                                              momentum_tauMu=momentum_tauMu)
        # inverse_A = find_inverse_of_dir_matrix(vector_plane_lb=vector_plane_lb, momentum_pkmu=vector_with_mu1,
        #                                             momentum_tauMu=momentum_tauMu, determinant=det)
        # coefficient_matrix = [[vector_plane_lb[j], vector_with_mu1[j], - momentum_tauMu[j]] for j in range(3)]
        # ordinate = [point_tau_mu[j] - end_xyz[j] for j in range(3)]
        # possible_calculation_u = sum([inverse_A[2][j] * ordinate[j] for j in range(3)])
        tau_distances_travelled.append(tau_distance)
        angle = np.arccos(np.dot(tau_vector, vector) / (np.linalg.norm(tau_vector) * np.linalg.norm(vector)))
        # angle_content = np.dot(tau_vector, vector) / (np.linalg.norm(tau_vector) * np.linalg.norm(vector))
        # bottom_content = (np.linalg.norm(tau_vector) * np.linalg.norm(vector))
        angles.append(angle)
        unit_l = vector / np.linalg.norm(vector)
        p_transverse = np.linalg.norm(temp_series['transverse_momentum'])
        tau_mom = p_transverse / np.tan(angle) * unit_l + temp_series['transverse_momentum']
        # sigma_tau = error_propagation_momentum_mass(data_frame=data_frame, inverse_A=inverse_A*det, determinant=det,
        #                                             i=i, angle=angle, unit_l=unit_l, p_transverse=p_transverse,
        #                                             bottom_content=bottom_content, angle_content=angle_content,
        #                                             possible_calculation_u=possible_calculation_u, ordinate=ordinate)
        # print(tau_mom, sigma_tau)

        tau_p_x.append(tau_mom[0])
        tau_p_y.append(tau_mom[1])
        tau_p_z.append(tau_mom[2])
        tau_p.append(np.linalg.norm(tau_mom))

    data_frame['tau_PX'] = tau_p_x
    data_frame['tau_PY'] = tau_p_y
    data_frame['tau_PZ'] = tau_p_z
    data_frame['tau_P'] = tau_p

    plt.hist(data_frame['tau_P'], bins=200, range=[0, 200000], density=True)
    plt.title('Momenta of the the "taus"')
    plt.show()
    data_frame['tau_distances_travelled'] = tau_distances_travelled
    plt.hist(data_frame['tau_distances_travelled'], bins=80, range=[0, 30])
    plt.title('Distance travelled by the "taus"')
    plt.show()

    # np.save('distance_real_taus_less.npy', data_frame['tau_distances_travelled'].values)
    # np.save('distance_real_taus.npy', data_frame['tau_distances_travelled'].values)
    # np.save('distance_jpsi_taus.npy', data_frame['tau_distances_travelled'].values)
    # np.save('bMC_below_5_tau_distance.npy', data_frame['tau_distances_travelled'].values)

    plt.hist2d(data_frame['tau_distances_travelled'], data_frame['lb_distances'], bins=20, range=[[0, 30], [0, 30]],
               norm=LogNorm())
    plt.title('Distance travelled by the "taus" versus lb')
    plt.colorbar()
    plt.show()

    # data_frame = data_frame[data_frame['tau_distances_travelled'] < 2]
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def mass(frame_array):
    return frame_array[0] ** 2 - (frame_array[1] ** 2 + frame_array[2] ** 2 + frame_array[3] ** 2)


def momentum(frame_array):
    mom = pd.concat([frame_array[1], frame_array[2], frame_array[3]], axis=1)
    mom.columns = ['X', 'Y', 'Z']
    return mom


def plot_b_result(data_frame):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['b_mass'] = sum_m

    # np.save('B_MC_mass.npy', data_frame['b_mass'].values)

    print('final length of the data', len(sum_m))
    n, b, p = plt.hist(sum_m, bins=100, range=[4000, 8000])
    plt.vlines(5279, ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{B}$')
    plt.ylabel('occurrences')
    plt.show()


def plot_result(data_frame):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['pkmutau_mass'] = sum_m
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pkmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    plt.hist2d(data_frame['pkmumu_mass'], data_frame['pkmutau_mass'], bins=20, range=[[2000, 8000], [4000, 8000]],
               norm=LogNorm())
    # plt.hist2d(data_frame['Lb_M'], sum_m, bins=30, range=[[0, 10000], [0, 20000]], norm=LogNorm())
    plt.axvline(masses['Lb'])
    plt.xlabel('$m_{pK\\mu\\mu}$')
    plt.ylabel('$m_{pK\\mu\\tau}$')
    plt.axhline(masses['Lb'])
    plt.colorbar()
    plt.show()

    # np.save('pkmutau_mass_cleaned.npy', data_frame['pkmutau_mass'].values)
    # np.save('pkmumu_mass_cleaned.npy', data_frame['pkmumu_mass'].values)

    print('final length of the data', len(sum_m))
    minimum_mass_pkmutau = masses['proton'] + masses['K'] + masses['mu'] + masses['tau']
    # n, b, p = plt.hist(data_frame['pkmutau_mass'], bins=100, range=[4000, 15000])
    n, b, p = plt.hist(data_frame['pkmutau_mass'], bins=100, range=[minimum_mass_pkmutau - 100, 15000])
    # n, b, p = plt.hist(sum_m, bins=100, range=[0, 20000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n))
    # plt.vlines(minimum_mass_pkmutau, ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{pK\\mu\\tau}$')
    plt.ylabel('occurrences')
    plt.show()
    minimum_mass_pkmumu = masses['proton'] + masses['K'] + masses['mu'] + masses['mu']
    # n, b, p = plt.hist(data_frame['pkmumu_mass'], bins=100, range=[5500, 5750])
    n, b, p = plt.hist(data_frame['pkmumu_mass'], bins=100, range=[minimum_mass_pkmumu - 100, 9000])
    # n, b, p = plt.hist(data_frame['Lb_M'], bins=100, range=[0, 10000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n))
    # plt.vlines(minimum_mass_pkmumu, ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{pK\\mu\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    new_frame = data_frame[data_frame['pkmutau_mass'] < 6500]
    new_frame = new_frame[new_frame['pkmutau_mass'] > 4500]
    plt.hist(new_frame['pKmu_ENDVERTEX_CHI2'], bins=50)
    plt.xlabel('$pK\\mu$' + ' endvertex  ' + '$\\chi^2$')
    plt.ylabel('occurrences')
    plt.show()
    plt.hist(new_frame['pKmu_IPCHI2_OWNPV'], bins=50)
    plt.xlabel('$pK\\mu$' + ' IP ' + '$\\chi^2$' + ' OWNPV')
    plt.ylabel('occurrences')
    plt.show()
    plt.hist(get_mass(data_frame=data_frame, particles_associations=[['mu1_P', 'mu'], ['tau_P', 'tau']]), bins=50)
    plt.xlabel('$m_{\\mu\\tau}$')
    plt.ylabel('occurrences')
    plt.show()
    plt.hist(get_mass(data_frame=data_frame, particles_associations=[['mu1_P', 'mu'], ['tauMu_P', 'mu']]), bins=50)
    plt.xlabel('$m_{\\mu\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    return


def get_missing_mass(data_frame):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    # lb_energy = np.sqrt(data_frame['Lb_P'] ** 2 + masses['Lb'] ** 2)
    missing_mass = masses['Lb'] - get_mass(data_frame=data_frame, particles_associations=particles_associations)
    print(missing_mass.describe())
    data_frame['missing_mass'] = missing_mass
    return data_frame


def get_dimuon_mass(data_frame):
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['dimuon_mass'] = sum_m
    return data_frame


if __name__ == '__main__':
    a = load_data(add_branches())
    a.dropna(inplace=True)
    df = reduce_background(a)
    # df = b_cleaning(a)
    # df = a
    df = df.reset_index(drop=True)
    df = get_dimuon_mass(df)
    df = retrieve_vertices(df)
    df = transverse_momentum(df)
    df = line_plane_intersection(df)
    df = tau_momentum_mass(df)
    plot_result(df)
    # plot_b_result(df)
