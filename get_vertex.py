from matplotlib.colors import LogNorm

from background_reduction import reduce_background, b_cleaning
from data_loader import load_data, add_branches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from masses import masses, get_mass
from scipy.special import factorial
import scipy.special as special
import scipy.optimize as spo

from error_propagation import error_propagation_intersection, error_propagation_transverse_momentum, \
    error_propagation_momentum_mass


def retrieve_vertices(data_frame):
    def exp_gaussian(x, l, mu, sigma):
        f = l / 2 * np.exp(l / 2 * (2 * mu + l * sigma ** 2 - 2 * x))
        complementary_error_function = special.erfc((mu + l * sigma ** 2 - x) / (np.sqrt(2) * sigma))
        return f * complementary_error_function

    x = np.linspace(0, 100, 200)
    n, b, p = plt.hist(data_frame['tauMu_P']/2000, bins=200, range=[0, 100], density=True)
    # print(spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.0001, 25000, 1000]))
    # (a, b, c), _ = spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.0001, 25000, 1000])
    # (a, b, c), _ = spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.1, 25, 10])
    # plt.plot(x, exp_gaussian(x, a, b, c), c='k')
    # plt.plot(x, exp_gaussian(x, 0.1, 25, 10), c='m')
    plt.title('Momenta of the the "taus"')
    plt.show()
    all_distances, vectors, errors = [], [], []
    print('initial length of data frame: ', len(data_frame))
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        pv_xyz = [ts['Lb_OWNPV_X'], ts['Lb_OWNPV_Y'], ts['Lb_OWNPV_Z']]
        end_xyz = [ts['pKmu_ENDVERTEX_X'], ts['pKmu_ENDVERTEX_Y'], ts['pKmu_ENDVERTEX_Z']]
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
    # compare_data = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3000)]
    # plt.hist(compare_data['lb_distances'], bins=100, range=[0, 100])
    # plt.show()
    # plt.hist(data_frame['lb_distances'], bins=100, range=[0, 100])
    # plt.title('Distances travelled by the Lb')
    # plt.show()
    data_frame['vectors'] = vectors
    # data_frame['vectors_errors'] = errors
    # data_frame = data_frame[(data_frame['lb_distances'] > 12)]
    # data_frame = data_frame.drop('lb_distances', axis=1)
    # data_frame = data_frame[data_frame['pKmu_IPCHI2_OWNPV'] > 9]
    # data_frame = data_frame[data_frame['pKmu_OWNPV_CHI2'] > 9]
    # data_frame = data_frame[data_frame['lb_distances'] > 5]
    # data_frame = data_frame[data_frame['lb_distances'] < 25]
    data_frame = data_frame.reset_index(drop=True)
    return data_frame, vectors


def line_plane_intersection(data_frame):
    intersections, muon_from_tau = [], []
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        end_xyz = [ts['pKmu_ENDVERTEX_X'], ts['pKmu_ENDVERTEX_Y'], ts['pKmu_ENDVERTEX_Z']]
        momentum_pkmu = [ts['pKmu_PX'], ts['pKmu_PY'], ts['pKmu_PZ']]
        momentum_tauMu = [ts['tauMu_PX'], ts['tauMu_PY'], ts['tauMu_PZ']]
        point_tau_mu = [ts['tauMu_REFPX'], ts['tauMu_REFPY'], ts['tauMu_REFPZ']]
        vector_plane_lb = ts['vectors']
        vector_with_mu1 = momentum_pkmu
        coefficient_matrix = [[vector_plane_lb[i], vector_with_mu1[i], - momentum_tauMu[i]] for i in range(3)]
        determinant = -vector_plane_lb[0] * vector_with_mu1[1] * momentum_tauMu[2] - momentum_tauMu[0] * \
                      vector_plane_lb[1] * vector_with_mu1[2] - vector_with_mu1[0] * momentum_tauMu[1] * \
                      vector_plane_lb[2] + momentum_tauMu[0] * vector_with_mu1[1] * vector_plane_lb[2] + \
                      vector_plane_lb[0] * momentum_tauMu[1] * vector_with_mu1[2] + vector_with_mu1[0] * \
                      vector_plane_lb[1] * momentum_tauMu[2]
        inverse_A = [[-momentum_tauMu[2] * vector_with_mu1[1] + momentum_tauMu[1] * vector_with_mu1[2],
                      -momentum_tauMu[0] * vector_with_mu1[2] + momentum_tauMu[2] * vector_with_mu1[0],
                      -momentum_tauMu[1] * vector_with_mu1[0] + momentum_tauMu[0] * vector_with_mu1[1]],
                     [-momentum_tauMu[1] * vector_plane_lb[2] + momentum_tauMu[2] * vector_plane_lb[1],
                      -momentum_tauMu[2] * vector_plane_lb[0] + momentum_tauMu[0] * vector_plane_lb[2],
                      -momentum_tauMu[0] * vector_plane_lb[1] + momentum_tauMu[1] * vector_plane_lb[0]],
                     [vector_plane_lb[1] * vector_with_mu1[2] - vector_plane_lb[2] * vector_with_mu1[1],
                      vector_plane_lb[2] * vector_with_mu1[0] - vector_plane_lb[0] * vector_with_mu1[2],
                      vector_plane_lb[0] * vector_with_mu1[1] - vector_plane_lb[1] * vector_with_mu1[0]]]
        if np.linalg.matrix_rank(coefficient_matrix) == np.array(coefficient_matrix).shape[0]:
            ordinate = [point_tau_mu[j] - end_xyz[j] for j in range(3)]
            # possible_intersection = np.linalg.solve(coefficient_matrix, ordinate)
            # possible_calculation_s = 1 / determinant * sum([inverse_A[0][j] * ordinate[j] for j in range(3)])
            # possible_calculation_t = 1 / determinant * sum([inverse_A[1][j] * ordinate[j] for j in range(3)])
            possible_calculation_u = 1 / determinant * sum([inverse_A[2][j] * ordinate[j] for j in range(3)])
            # print(possible_intersection)
            # print([possible_calculation_s, possible_calculation_t, possible_calculation_u])
            possible_intersection = np.array(momentum_tauMu) * possible_calculation_u + np.array(point_tau_mu)
            # print(possible_intersection)
            # sigma_i = error_propagation_intersection(data_frame=data_frame, inverse_A=inverse_A,
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


def transverse_momentum(data_frame, vectors):
    pkmu = [np.sqrt(data_frame['pKmu_P'] ** 2), data_frame['pKmu_PX'], data_frame['pKmu_PY'], data_frame['pKmu_PZ']]
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
        tau_vector = temp_series['tau_decay_point'] - end_xyz
        vector = temp_series['vectors']
        tau_distance = np.linalg.norm(tau_vector)  # error on tau distance not needed for now
        momentum_tauMu = [temp_series['tauMu_PX'], temp_series['tauMu_PY'], temp_series['tauMu_PZ']]
        pkmu_vector = [temp_series['pKmu_PX'], temp_series['pKmu_PY'], temp_series['pKmu_PZ']]
        vector_plane_lb = temp_series['vectors']
        vector_with_mu1 = [temp_series['pKmu_PX'], temp_series['pKmu_PY'], temp_series['pKmu_PZ']]
        point_tau_mu = [temp_series['tauMu_REFPX'], temp_series['tauMu_REFPY'], temp_series['tauMu_REFPZ']]
        determinant = -vector_plane_lb[0] * vector_with_mu1[1] * momentum_tauMu[2] - momentum_tauMu[0] * \
                      vector_plane_lb[1] * vector_with_mu1[2] - vector_with_mu1[0] * momentum_tauMu[1] * \
                      vector_plane_lb[2] + momentum_tauMu[0] * vector_with_mu1[1] * vector_plane_lb[2] + \
                      vector_plane_lb[0] * momentum_tauMu[1] * vector_with_mu1[2] + vector_with_mu1[0] * \
                      vector_plane_lb[1] * momentum_tauMu[2]
        inverse_A = [[-momentum_tauMu[2] * vector_with_mu1[1] + momentum_tauMu[1] * vector_with_mu1[2],
                      -momentum_tauMu[0] * vector_with_mu1[2] + momentum_tauMu[2] * vector_with_mu1[0],
                      -momentum_tauMu[1] * vector_with_mu1[0] + momentum_tauMu[0] * vector_with_mu1[1]],
                     [-momentum_tauMu[1] * vector_plane_lb[2] + momentum_tauMu[2] * vector_plane_lb[1],
                      -momentum_tauMu[2] * vector_plane_lb[0] + momentum_tauMu[0] * vector_plane_lb[2],
                      -momentum_tauMu[0] * vector_plane_lb[1] + momentum_tauMu[1] * vector_plane_lb[0]],
                     [vector_plane_lb[1] * vector_with_mu1[2] - vector_plane_lb[2] * vector_with_mu1[1],
                      vector_plane_lb[2] * vector_with_mu1[0] - vector_plane_lb[0] * vector_with_mu1[2],
                      vector_plane_lb[0] * vector_with_mu1[1] - vector_plane_lb[1] * vector_with_mu1[0]]]
        coefficient_matrix = [[vector_plane_lb[j], vector_with_mu1[j], - momentum_tauMu[j]] for j in range(3)]
        ordinate = [point_tau_mu[j] - end_xyz[j] for j in range(3)]
        possible_calculation_u = 1 / determinant * sum([inverse_A[2][j] * ordinate[j] for j in range(3)])
        tau_distances_travelled.append(tau_distance)
        angle = np.arccos(np.dot(tau_vector, vector) / (np.linalg.norm(tau_vector) * np.linalg.norm(vector)))
        angle_content = np.dot(tau_vector, vector) / (np.linalg.norm(tau_vector) * np.linalg.norm(vector))
        bottom_content = (np.linalg.norm(tau_vector) * np.linalg.norm(vector))
        top_content = np.dot(tau_vector, vector)
        angles.append(angle)
        unit_l = vector / np.linalg.norm(vector)
        p_transverse = np.linalg.norm(temp_series['transverse_momentum'])
        tau_mom = p_transverse / np.tan(angle) * unit_l + temp_series['transverse_momentum']
        tau_par = p_transverse / np.tan(angle) * unit_l  # some stuff will just need tau par
        # sigma_tau = error_propagation_momentum_mass(data_frame=data_frame, inverse_A=inverse_A, determinant=determinant,
        #                                             i=i, angle=angle, unit_l=unit_l, p_transverse=p_transverse,
        #                                             bottom_content=bottom_content, angle_content=angle_content,
        #                                             possible_calculation_u=possible_calculation_u, ordinate=ordinate)
        # print(tau_mom, sigma_tau)
        # print(dt_x_dp_k_mu_ref_x, dt_x_dp_k_mu_ref_y, dt_x_dp_k_mu_ref_z)
        # print(dt_x_d_tau_ref_x, dt_x_d_tau_ref_y, dt_x_d_tau_ref_z)
        # print(dt_x_d_tau_mom_x, dt_x_d_tau_mom_y, dt_x_d_tau_mom_z)
        # print(dt_x_d_vector_lb_x, dt_x_d_vector_lb_y, dt_x_d_vector_lb_z)
        # print(dt_x_dp_k_mu_mom_x, dt_x_dp_k_mu_mom_y, dt_x_dp_k_mu_mom_z)

        tau_p_x.append(tau_mom[0])
        tau_p_y.append(tau_mom[1])
        tau_p_z.append(tau_mom[2])
        tau_p.append(np.linalg.norm(tau_mom))

    data_frame['tau_PX'] = tau_p_x
    data_frame['tau_PY'] = tau_p_y
    data_frame['tau_PZ'] = tau_p_z
    data_frame['tau_P'] = tau_p

    def exp_gaussian(x, l, mu, sigma):
        f = l / 2 * np.exp(l / 2 * (2 * mu + l * sigma ** 2 - 2 * x))
        complementary_error_function = special.erfc((mu + l * sigma ** 2 - x) / (np.sqrt(2) * sigma))
        return f * complementary_error_function

    x = np.linspace(0, 200000, 200)
    n, b, p = plt.hist(data_frame['tau_P'], bins=200, range=[0, 200000], density=True)
    # print(spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.0001, 25000, 1000]))
    # (a, b, c), _ = spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.0001, 25000, 1000])
    # (a, b, c), _ = spo.curve_fit(exp_gaussian, ((b[1:] + b[:-1]) / 2), n, p0=[0.1, 25, 10])
    # plt.plot(x, exp_gaussian(x, a, b, c), c='k')
    # plt.plot(x, exp_gaussian(x, 0.1, 25, 10), c='m')
    plt.title('Momenta of the the "taus"')
    plt.show()
    data_frame['tau_distances_travelled'] = tau_distances_travelled
    compare_data = data_frame[(data_frame['dimuon_mass'] < 3150) & (data_frame['dimuon_mass'] > 3000)]
    # plt.hist(compare_data['tau_distances_travelled'], bins=100, range=[0, 200])
    plt.hist(data_frame['tau_distances_travelled'], bins=80, range=[0, 30])
    plt.title('Distance travelled by the "taus"')
    plt.show()

    # np.save('distance_real_taus.npy', data_frame['tau_distances_travelled'].values)

    plt.hist2d(data_frame['tau_distances_travelled'], data_frame['lb_distances'], bins=20, range=[[0, 30], [0, 30]], norm=LogNorm())
    plt.title('Distance travelled by the "taus" versus lb')
    plt.colorbar()
    plt.show()

    # data_frame = data_frame[data_frame['tau_distances_travelled'] < 15]
    # data_frame = data_frame[data_frame['tau_distances_travelled'] > 0.3]
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

    # TODO add error for sum_m
    print('final length of the data', len(sum_m))
    n, b, p = plt.hist(sum_m, bins=60, range=[3500, 10000])
    plt.vlines(5279, ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{B}$')
    plt.ylabel('occurrences')
    plt.show()
    n, b, p = plt.hist(data_frame['Lb_M'], bins=60, range=[0, 10000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n))
    plt.vlines(5279, ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{lb}$')
    plt.ylabel('occurrences')
    plt.show()


def plot_result(data_frame):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    # TODO add errors for moms and energy
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['pkmutau_mass'] = sum_m
    # particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    # TODO add errors for moms and energy
    data_frame['pkmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    plt.hist2d(data_frame['pkmumu_mass'], data_frame['pkmutau_mass'], bins=20, range=[[2000, 8000], [4000, 8000]], norm=LogNorm())
    plt.axvline(masses['Lb'])
    plt.xlabel('$m_{pK\\mu\\mu}$')
    plt.ylabel('$m_{pK\\mu\\tau}$')
    plt.axhline(masses['Lb'])
    plt.colorbar()
    plt.show()

    # plt.hist2d(data_frame['kmu_mass'], data_frame['pkmutau_mass'], bins=20, range=[[0, 4000], [4000, 8000]], norm=LogNorm())
    # plt.axhline(masses['Lb'])
    # plt.colorbar()
    # plt.show()

    # TODO add error for sum_m
    print('final length of the data', len(sum_m))
    # plt.hist(sum_m, bins=50, range=[4500, 6500])
    n, b, p = plt.hist(sum_m, bins=60, range=[4000, 15000])
    # n, b, p = plt.hist(data_frame['Lb_M'], bins=100, range=[0, 10000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{pK\\mu\\tau}$')
    plt.ylabel('occurrences')
    plt.show()
    n, b, p = plt.hist(data_frame['Lb_M'], bins=60, range=[4000, 6000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n))
    plt.xlabel('$m_{pK\\mu\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    new_frame = data_frame[data_frame['pkmutau_mass'] < 6500]
    new_frame = new_frame[new_frame['pkmutau_mass'] > 4500]
    new_frame.dropna(inplace=True)
    plt.hist(new_frame['pKmu_ENDVERTEX_CHI2'], bins=50)
    plt.xlabel('$pK\\mu$' + ' endvertex  ' + '$\\chi^2$')
    plt.ylabel('occurrences')
    plt.show()
    plt.hist(new_frame['pKmu_IPCHI2_OWNPV'], bins=50)
    plt.xlabel('$pK\\mu$' + ' IP ' + '$\\chi^2$' + ' OWNPV')
    plt.ylabel('occurrences')
    plt.show()
    particles_associations = [['mu1_P', 'mu'], ['tau_P', 'tau']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)

    plt.hist(sum_m, bins=50)
    # need to take cuts around those masses
    plt.xlabel('$m_{\\mu\\tau}$')
    plt.ylabel('occurrences')
    plt.show()
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    plt.hist(sum_m, bins=50)
    # need to take cuts around those masses
    plt.xlabel('$m_{\\mu\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    return sum_m


def get_missing_mass(data_frame):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    lb_energy = np.sqrt(data_frame['Lb_P'] ** 2 + masses['Lb'] ** 2)
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
    # df = reduce_background(a)
    # df = df[df['Lb_pmu_ISOLATION_BDT1'] == -2]
    # df = df[df['Lb_pmu_ISOLATION_BDT1'] < 0]
    # df = df.reset_index(drop=True)
    df = b_cleaning(a)
    # plt.hist(df['pKmu_PT'], bins=50)
    # plt.xlabel('pKmu_PT')
    # plt.ylabel('occurrences')
    # plt.show()
    # plt.hist(df['mu1_PT'], bins=50)
    # plt.xlabel('mu1_PT')
    # plt.ylabel('occurrences')
    # plt.show()
    # df = df[df['pKmu_PT'] > 5000]
    # df = df[df['mu1_PT'] > 2000]
    # df = df.reset_index(drop=True)
    df = get_dimuon_mass(df)
    df = get_missing_mass(df)
    df, vec = retrieve_vertices(df)
    df = get_missing_mass(df)
    df = transverse_momentum(df, vec)
    df = line_plane_intersection(df)
    df = tau_momentum_mass(df)
    # plot_result(df)
    plot_b_result(df)
