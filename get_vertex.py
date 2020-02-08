import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from background_reduction import reduce_background, b_cleaning
from data_loader import load_data, add_branches
from ip_calculations import line_line_intersection, line_point_distance, return_all_ip
from masses import masses, get_mass
from error_propagation import error_propagation_intersection, error_propagation_transverse_momentum, \
    error_propagation_momentum_mass
from matrix_calculations import find_determinant_of_dir_matrix, find_inverse_of_dir_matrix


def retrieve_vertices(data_frame):
    """
    Obtains the Lambda B line of flight and flight distance
    :param data_frame:
    :return:
    """
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
        # errors_pv = [ts['Lb_OWNPV_XERR'], ts['Lb_OWNPV_YERR'], ts['Lb_OWNPV_ZERR']]
        # errors_end = [ts['pKmu_REFP_COVXX'], ts['pKmu_REFP_COVYY'], ts['pKmu_REFP_COVZZ']]
        distance = np.linalg.norm(np.array(pv_xyz) - np.array(end_xyz))
        vector = np.array(end_xyz) - np.array(pv_xyz)
        # errors.append([np.sqrt((errors_pv[i]) ** 2 + (errors_end[i])) for i in range(3)])
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
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def line_plane_intersection(data_frame):
    """
    Finds intersection between the tauMu, and the plane made of the Lb line of light and pkmu momentum vector
    :param data_frame:
    :return:
    """
    intersections, muon_from_tau = [], []
    angles_mu = []
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
        # now find angle between tauMu and the plane
        normal_to_plane = np.cross(vector_plane_lb, momentum_pkmu)
        angle_mu_plane_normal = np.arccos(np.dot(normal_to_plane, momentum_tauMu) / (
                np.linalg.norm(normal_to_plane) * np.linalg.norm(momentum_tauMu)))
        angle_mu_plane_normal = np.degrees(angle_mu_plane_normal)
        angle_mu_plane = 90 - angle_mu_plane_normal if angle_mu_plane_normal < 90 else angle_mu_plane_normal - 90
        angles_mu.append(angle_mu_plane)

    data_frame['muon_from_tau'] = muon_from_tau
    data_frame['tau_decay_point'] = intersections
    data_frame = data_frame[data_frame['muon_from_tau'] > 0]
    data_frame['angle_to_plane'] = angles_mu
    # data_frame = data_frame[data_frame['angle_to_plane'] > 0.3]
    data_frame = data_frame.reset_index(drop=True)
    # np.save('angle_cleaned_data.npy', angles_mu)
    plt.hist(angles_mu, bins=100, range=[0, 4])
    plt.xlabel('Angle between tauMu and the plane')
    plt.ylabel('Occurrences')
    plt.show()
    print((len(data_frame)))

    return data_frame


def transverse_momentum(data_frame):
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
        # t_mom_err = error_propagation_transverse_momentum(data_frame=data_frame, vectors=vectors, i=i,
        #                                                   pkmu_vector=pkmu_vector)
        transverse_momenta.append(-transverse_mom)
    data_frame['transverse_momentum'] = transverse_momenta
    return data_frame


def tau_momentum_mass(data_frame):
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
        tau_distances_travelled.append(tau_distance * np.sign(np.dot(tau_vector, vector)))
        # tau_distances_travelled.append(tau_distance)
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
    plt.hist(data_frame['tau_distances_travelled'], bins=80, range=[0, 15])
    plt.title('Distance travelled by the "taus"')
    plt.show()

    # np.save('distance_taus_cleaned_data.npy', data_frame['tau_distances_travelled'].values)
    # np.save('distance_taus_pkmu_below_2300.npy', data_frame['tau_distances_travelled'].values)
    # np.save('distance_real_taus.npy', data_frame['tau_distances_travelled'].values)
    # np.save('distance_jpsi_taus.npy', data_frame['tau_distances_travelled'].values)

    plt.hist2d(data_frame['tau_distances_travelled'], data_frame['lb_distances'], bins=20, range=[[0, 30], [0, 30]],
               norm=LogNorm())
    plt.title('Distance travelled by the "taus" versus lb')
    plt.colorbar()
    plt.show()

    # data_frame = data_frame[data_frame['Lb_M'] < 5520]
    # data_frame = data_frame.reset_index(drop=True)
    data_frame['vector_muTau'] = data_frame[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    data_frame['tauMu_reference_point'] = data_frame[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    data_frame['pkmu_endvertex_point'] = data_frame[
        ['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    data_frame['pkmu_direction'] = data_frame[['pKmu_PX', 'pKmu_PY', 'pKmu_PZ']].values.tolist()
    data_frame['impact_parameter_thingy'] = line_point_distance(vector=data_frame['vector_muTau'],
                                                                vector_point=data_frame['tauMu_reference_point'],
                                                                point=data_frame['pkmu_endvertex_point'],
                                                                direction=data_frame['pkmu_direction'])
    plt.hist(data_frame['impact_parameter_thingy'], range=[-0.4, 0.4], bins=100)
    plt.ylabel('occurrences')
    plt.xlabel('IP*')
    plt.show()
    # np.save('ipstar_250_cleaned_data_nolb.npy', data_frame['impact_parameter_thingy'].values)

    plt.hist2d(data_frame['impact_parameter_thingy'], data_frame['tau_distances_travelled'],
               range=[[-0.1, 0.1], [-7.5, 7.5]], bins=50, norm=LogNorm())
    plt.show()
    # data_frame = data_frame[data_frame['tau_distances_travelled'] > -1]
    # plt.hist(data_frame['pkmu_mass'], bins=100)
    # plt.xlabel('pkmu mass')
    # plt.savefig('pkmu_mass_all.png')
    data_frame, ip_cols = return_all_ip(data_frame)
    # data_frame = data_frame[data_frame['Lb_M'] < 5520]
    # data_frame = data_frame.reset_index(drop=True)
    # for col in ip_cols:
    #     plt.hist(data_frame[col], range=[-0.4, 0.4], bins=100)
    #     plt.ylabel('occurrences')
    #     plt.xlabel(col)
    #     plt.show()
        # np.save(col + 'background_cleaned_data.npy', data_frame[col].values)

    # plt.show()
    # new_data = data_frame[data_frame['tau_distances_travelled'] < -0]
    # plt.hist(new_data['pkmu_mass'], bins=100)
    # plt.xlabel('pkmu mass below a tau distance of 0mm')
    # plt.savefig('pkmu_mass_tau_dist_below_0.png')
    # plt.show()
    # np.save('ipstar_jpsi_sign.npy', data_frame['impact_parameter_thingy'].values)
    # data_frame = data_frame[data_frame['impact_parameter_thingy'] > -0.1]
    # data_frame = data_frame[data_frame['impact_parameter_thingy'] < 0.1]
    data_frame = data_frame[data_frame['impact_parameter_thingy'] > -0.02]
    # data_frame = data_frame[data_frame['pk_ip'] > 0]
    # data_frame = data_frame[data_frame['pk_iptau'] > -0.15]
    # data_frame = data_frame[data_frame['impact_parameter_thingy'] < 0.02]
    # data_frame = data_frame[data_frame['impact_parameter_thingy'] > -0.02]
    # data_frame = data_frame[data_frame['tau_distances_travelled'] < 2]
    # data_frame = data_frame[(data_frame['tau_distances_travelled'] > 0)]
    # data_frame = data_frame[(data_frame['tau_distances_travelled'] > 0) & (data_frame['tau_distances_travelled'] < 4)]
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def momentum(frame_array):
    """
    Returns momentum of data frame made of [energy, Px Py, Pz]
    :param frame_array:
    :return:
    """
    mom = pd.concat([frame_array[1], frame_array[2], frame_array[3]], axis=1)
    mom.columns = ['X', 'Y', 'Z']
    return mom


def plot_b_result(data_frame):
    """
    Plots mass results for the B particles
    :param data_frame:
    :return:
    """
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
    """
    Plots mass results for the Lb particles
    :param data_frame:
    :return:
    """
    # particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tau_P', 'pi']]
    # data_frame['pkmupi_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    #
    # n, b, p = plt.hist(data_frame['pkmupi_mass'], bins=100, range=[2000, 14000])
    # plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n), linewidth=0.5)
    # plt.xlabel('$m_{pK\\mu\\pi}$')
    # plt.ylabel('occurrences')
    # plt.show()
    # data_frame = data_frame[(data_frame['pkmupi_mass'] < masses['Lb']-150) | (data_frame['pkmupi_mass'] > masses['Lb']+150)]
    # data_frame = data_frame[(data_frame['Lb_M'] < 5520) | (data_frame['Lb_M'] > 5720)]

    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pkmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    length_pkmumu_signal = len(
        data_frame[(data_frame['pkmumu_mass'] > 5620 - 40) & (data_frame['pkmumu_mass'] < 5620 + 40)])
    print(len(data_frame), length_pkmumu_signal / len(data_frame))
    minimum_mass_pkmumu = masses['proton'] + masses['K'] + masses['mu'] + masses['mu']
    # n, b, p = plt.hist(data_frame['pkmumu_mass'], bins=100, range=[5500, 5750])
    # plt.figure(figsize=(3.5, 3), dpi=300)
    n, b, p = plt.hist(data_frame['pkmumu_mass'], bins=100, range=[minimum_mass_pkmumu - 100, 9000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n), linewidth=0.5)
    plt.xlabel('$m_{pK\\mu\\mu}$')
    plt.ylabel('occurrences')
    # plt.tight_layout()
    plt.show()
    # comparison_data = data_frame[(data_frame['pkmumu_mass'] > 5620 - 40) & (data_frame['pkmumu_mass'] < 5620 + 40)]
    # plt.hist(comparison_data['pk_mass'], bins=50)
    # plt.xlabel('pk_mass')
    # plt.show()

    # for tau_masses in [50*x for x in range(36)]:
    #     particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    #     data_frame['pkmutau_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations, ms=[['tau', tau_masses]])
    #     # np.save('pkmutau_mass_cleaned.npy', data_frame['pkmutau_mass'].values)
    #     # np.save('pkmumu_mass_cleaned.npy', data_frame['pkmumu_mass'].values)
    #     print('final length of the data', len(data_frame))
    #     minimum_mass_pkmutau = masses['proton'] + masses['K'] + masses['mu'] + tau_masses
    #     # n, b, p = plt.hist(data_frame['pkmutau_mass'], bins=100, range=[4000, 15000])
    #     n, b, p = plt.hist(data_frame['pkmutau_mass'], bins=100, range=[minimum_mass_pkmutau - 100, 15000])
    #     plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n))
    #     plt.xlabel('$m_{pK\\mu X}$ with $m_{X}=$' + str(tau_masses) + 'MeV')
    #     plt.ylabel('occurrences')
    #     # plt.show()
    #     plt.savefig(f'pkmuX_mass_{tau_masses}.png')
    #     plt.close('all')

    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    data_frame['pkmutau_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    # np.save('pkmutau_mass_cleaned.npy', data_frame['pkmutau_mass'].values)
    # np.save('pkmumu_mass_cleaned.npy', data_frame['pkmumu_mass'].values)
    print('final length of the data', len(data_frame))
    minimum_mass_pkmutau = masses['proton'] + masses['K'] + masses['mu'] + masses['tau']
    # n, b, p = plt.hist(data_frame['pkmutau_mass'], bins=100, range=[4000, 15000])
    # plt.figure(figsize=(3.5, 3), dpi=300)
    n, b, p = plt.hist(data_frame['pkmutau_mass'], bins=100, range=[minimum_mass_pkmutau - 100, 15000])
    plt.vlines(masses['Lb'], ymin=0, ymax=np.max(n), linewidth=0.5)
    plt.xlabel('$m_{pK\\mu\\tau}$')
    plt.ylabel('occurrences')
    # plt.tight_layout()
    plt.show()
    plt.hist2d(data_frame['pkmumu_mass'], data_frame['pkmutau_mass'], bins=20, range=[[2000, 8000], [4000, 8000]],
               norm=LogNorm())
    # plt.hist2d(data_frame['Lb_M'], sum_m, bins=30, range=[[0, 10000], [0, 20000]], norm=LogNorm())
    plt.axvline(masses['Lb'])
    plt.xlabel('$m_{pK\\mu\\mu}$')
    plt.ylabel('$m_{pK\\mu\\tau}$')
    plt.axhline(masses['Lb'])
    plt.colorbar()
    plt.show()

    new_frame = data_frame[(data_frame['pkmutau_mass'] < 6500) & (data_frame['pkmutau_mass'] > 4500)]
    plt.hist(new_frame['pKmu_ENDVERTEX_CHI2'], bins=50)
    plt.xlabel('pKmu_ENDVERTEX_CHI2')
    plt.ylabel('occurrences')
    plt.show()
    plt.hist(new_frame['pKmu_IPCHI2_OWNPV'], bins=50)
    plt.xlabel('pKmu_IPCHI2_OWNPV')
    plt.ylabel('occurrences')
    plt.show()
    plt.hist(get_mass(data_frame=data_frame, particles_associations=[['mu1_P', 'mu'], ['tau_P', 'tau']]),
             range=[1500, 7000], bins=50)
    plt.xlabel('$m_{\\mu\\tau}$')
    plt.ylabel('occurrences')
    plt.show()
    plt.hist(get_mass(data_frame=data_frame, particles_associations=[['mu1_P', 'mu'], ['tauMu_P', 'mu']]), range=[0, 6000], bins=120)
    plt.xlabel('$m_{\\mu\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    return


def get_dimuon_mass(data_frame):
    """
    Obtains the dimuon mass of the events
    :param data_frame:
    :return:
    """
    particles_associations = [['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    sum_m = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['dimuon_mass'] = sum_m
    return data_frame


if __name__ == '__main__':
    a = load_data(add_branches())
    a.dropna(inplace=True)
    df = reduce_background(a)
    # plt.hist(df['dimuon_mass']**2, bins=100, range=[0, 1.5e7])
    # plt.show()
    # df = df[df['dimuon_mass']**2 < 6e6]
    # df = df[df['dimuon_mass']**2 > 0.1e6]
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
    # plt.hist2d(df['proton_P'], df['proton_PIDp'], bins=50)
    # plt.xlabel('proton_P')
    # plt.ylabel('proton_PIDp')
    # plt.show()
    # plt.hist2d(df['Kminus_P'], df['Kminus_PIDK'], bins=50)
    # plt.xlabel('Kminus_P')
    # plt.ylabel('Kminus_PIDK')
    # plt.show()
    # df = b_cleaning(a)
    # df = a
    # df['vector_muTau'] = df[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    # df['tauMu_reference_point'] = df[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    # df['pkmu_endvertex_point'] = df[['pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z']].values.tolist()
    # df['impact_parameter_thingy'] = line_point_distance(vector=df['vector_muTau'], vector_point=df['tauMu_reference_point'],
    #                                     point=df['pkmu_endvertex_point'])
    # np.save('ipstar_bmc.npy', df['impact_parameter_thingy'].values)
    # data_frame = data_frame[data_frame['impact_parameter_thingy'] > 0.05]
    df = df.reset_index(drop=True)
    df = get_dimuon_mass(df)
    df = retrieve_vertices(df)
    df = transverse_momentum(df)
    df = line_plane_intersection(df)
    df = tau_momentum_mass(df)
    plot_result(df)
    # plot_b_result(df)
