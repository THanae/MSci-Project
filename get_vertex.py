from data_loader import load_data, add_branches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

masses = {'mu': 105.658, 'proton': 938.272, 'K': 493.677, 'pi': 139.57, 'Lb': 5260, 'tau': 1777}


def retrieve_vertices(data_frame):
    all_distances, vectors = [], []
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        pv_xyz = [ts['Lb_OWNPV_X'], ts['Lb_OWNPV_Y'], ts['Lb_OWNPV_Z']]
        end_xyz = [ts['Lb_ENDVERTEX_X'], ts['Lb_ENDVERTEX_Y'], ts['Lb_ENDVERTEX_Z']]
        # errors_pv = [ts['Lb_OWNPV_XERR'], ts['Lb_OWNPV_YERR'], ts['Lb_OWNPV_ZERR']]
        # errors_end = [ts['Lb_ENDVERTEX_XERR'], ts['Lb_ENDVERTEX_YERR'], ts['Lb_ENDVERTEX_ZERR']]
        distance = np.linalg.norm(np.array(pv_xyz) - np.array(end_xyz))
        vector = np.array(end_xyz) - np.array(pv_xyz)
        vectors.append(vector)
        all_distances.append(distance)
    data_frame['distances'] = all_distances
    data_frame['vectors'] = vectors
    data_frame = data_frame[data_frame['distances'] > 18]  # should be changed according to what we want
    data_frame = data_frame[(data_frame['missing_mass1'] > masses['tau'] - masses['mu'])]
    data_frame = data_frame[data_frame['missing_mass2'] > masses['tau'] - masses['mu']]
    data_frame = data_frame.drop('distances', axis=1)
    data_frame = data_frame.reset_index()
    return data_frame, vectors


def line_plane_intersection(data_frame):
    intersections, muon_from_tau = [], []
    for i in range(len(data_frame)):
        end_xyz = [data_frame['Lb_ENDVERTEX_X'][i], data_frame['Lb_ENDVERTEX_Y'][i], data_frame['Lb_ENDVERTEX_Z'][i]]
        momentum_k = [data_frame['Kminus_PX'][i], data_frame['Kminus_PY'][i], data_frame['Kminus_PZ'][i]]
        momentum_p = [data_frame['proton_PX'][i], data_frame['proton_PY'][i], data_frame['proton_PZ'][i]]
        momentum_mu1 = [data_frame['mu1_PX'][i], data_frame['mu1_PY'][i], data_frame['mu1_PZ'][i]]
        momentum_mu2 = [data_frame['mu2_PX'][i], data_frame['mu2_PY'][i], data_frame['mu2_PZ'][i]]
        point_muon_1 = list(np.array(end_xyz) + np.random.uniform(low=1, high=5, size=(3,)))  # need to fill in
        point_muon_2 = list(np.array(end_xyz) + np.random.uniform(low=1, high=5, size=(3,)))  # need to fill in

        # whole plane definition thing
        # plane = end_xyz + number * vec + number * vec
        # line1/2 =  point_muon_1/2 + number * momentum_mu1/2
        vector_plane_lb = data_frame['vectors'][i]
        vector_with_mu1 = list(np.array(momentum_k) + np.array(momentum_p) + np.array(momentum_mu1))
        vector_with_mu2 = list(np.array(momentum_k) + np.array(momentum_p) + np.array(momentum_mu2))
        # normal_to_plane1 = np.cross(vector_plane_lb, vector_with_mu1)
        # normal_to_plane2 = np.cross(vector_plane_lb, vector_with_mu2)
        # angle_line_plane1 = np.arcsin(
        #     np.dot(normal_to_plane1, momentum_mu2) / (np.linalg.norm(normal_to_plane1) * np.linalg.norm(momentum_mu2)))
        # angle_line_plane2 = np.arcsin(
        #     np.dot(normal_to_plane2, momentum_mu1) / (np.linalg.norm(normal_to_plane2) * np.linalg.norm(momentum_mu1)))

        coefficient_matrix1 = [[vector_plane_lb[i], vector_with_mu2[i], - momentum_mu1[i]] for i in range(3)]
        coefficient_matrix2 = [[vector_plane_lb[i], vector_with_mu1[i], - momentum_mu2[i]] for i in range(3)]
        if np.linalg.matrix_rank(coefficient_matrix1) == np.array(coefficient_matrix1).shape[0]:
            ordinate1 = [end_xyz[i] - point_muon_1[i] for i in range(3)]
            possible_intersection1 = np.linalg.solve(coefficient_matrix1, ordinate1)
        else:
            possible_intersection1 = False
        if np.linalg.matrix_rank(coefficient_matrix2) == np.array(coefficient_matrix2).shape[0]:
            ordinate2 = [end_xyz[i] - point_muon_2[i] for i in range(3)]
            possible_intersection2 = np.linalg.solve(coefficient_matrix1, ordinate2)
        else:
            possible_intersection2 = False
        if possible_intersection1 is not False:
            intersection = possible_intersection1
            muon_from_tau.append(1)  # 0 for none, 1 for muon 1 and 2 for muon 2
        elif possible_intersection2 is not False:
            intersection = possible_intersection2
            muon_from_tau.append(2)  # 0 for none, 1 for muon 1 and 2 for muon 2
        else:
            muon_from_tau.append(0)  # 0 for none, 1 for muon 1 and 2 for muon 2
            intersection = [0, 0, 0]
        intersections.append(intersection)
    data_frame['muon_from_tau'] = muon_from_tau
    data_frame['tau_decay_point'] = intersections
    data_frame = data_frame[data_frame['muon_from_tau'] > 0]
    data_frame = data_frame.reset_index()
    return data_frame


def transverse_momentum(data_frame, vectors):
    k_minus = [np.sqrt(data_frame['Kminus_P'] ** 2 + masses['K'] ** 2), data_frame['Kminus_PX'],
               data_frame['Kminus_PY'], data_frame['Kminus_PZ']]
    proton = [np.sqrt(data_frame['proton_P'] ** 2 + masses['proton'] ** 2), data_frame['proton_PX'],
              data_frame['proton_PY'], data_frame['proton_PZ']]
    muon_1 = [np.sqrt(data_frame['mu1_P'] ** 2 + masses['mu'] ** 2), data_frame['mu1_PX'], data_frame['mu1_PY'],
              data_frame['mu1_PZ']]
    muon_2 = [np.sqrt(data_frame['mu2_P'] ** 2 + masses['mu'] ** 2), data_frame['mu2_PX'], data_frame['mu2_PY'],
              data_frame['mu2_PZ']]
    transverse_momenta1, transverse_momenta2 = [], []
    k_momentum, p_momentum = momentum(k_minus), momentum(proton)
    m1_momentum, m2_momentum = momentum(muon_1), momentum(muon_2)
    for i in range(len(data_frame)):
        par_vector = vectors[i]
        k_vector, p_vector = np.array(k_momentum.loc[i]), np.array(p_momentum.loc[i])
        m1_vector, m2_vector = np.array(m1_momentum.loc[i]), np.array(m2_momentum.loc[i])
        par_k = np.dot(k_vector, par_vector) * k_vector / np.linalg.norm(k_vector)
        par_p = np.dot(p_vector, par_vector) * p_vector / np.linalg.norm(p_vector)
        par_kp = par_k + par_p
        par1 = par_kp + np.dot(m1_vector, par_vector) * m1_vector / np.linalg.norm(m1_vector)
        par2 = par_kp + np.dot(m2_vector, par_vector) * m2_vector / np.linalg.norm(m2_vector)
        transverse_momentum1 = k_vector + p_vector + m1_vector - par1
        transverse_momentum2 = k_vector + p_vector + m2_vector - par2
        transverse_momenta1.append(-transverse_momentum1)
        transverse_momenta2.append(-transverse_momentum2)
    data_frame['transverse_momentum1'], data_frame['transverse_momentum2'] = transverse_momenta1, transverse_momenta2
    return data_frame, transverse_momenta1, transverse_momenta2


def tau_momentum_mass(data_frame):
    angles, tau_p, tau_distances_travelled = [], [], []
    tau_p_x, tau_p_y, tau_p_z = [], [], []
    for i in range(len(data_frame)):
        temp_series = data_frame.loc[i]
        end_xyz = [temp_series['Lb_ENDVERTEX_X'], temp_series['Lb_ENDVERTEX_Y'], temp_series['Lb_ENDVERTEX_Z']]
        tau_vector = temp_series['tau_decay_point'] - end_xyz
        vector = temp_series['vectors']
        tau_distance = np.linalg.norm(tau_vector)
        tau_distances_travelled.append(tau_distance)
        angle = np.arccos(np.dot(tau_vector, vector) / (np.linalg.norm(tau_vector) * np.linalg.norm(vector)))
        # angle = np.arctan((np.linalg.norm(tau_vector) * np.linalg.norm(vector))/np.dot(tau_vector, vector))
        print(angle, 1 / np.tan(angle))
        angles.append(angle)
        unit_l = vector / np.linalg.norm(vector)
        if temp_series['muon_from_tau'] == 1:
            p_tansverse = np.linalg.norm(temp_series['transverse_momentum2'])
            print(p_tansverse)
            tau_mom = p_tansverse / np.tan(angle) * unit_l + temp_series['transverse_momentum2']
        else:

            p_tansverse = np.linalg.norm(temp_series['transverse_momentum1'])
            print(p_tansverse)
            tau_mom = p_tansverse / np.tan(angle) * unit_l + temp_series['transverse_momentum1']
        print(tau_mom)
        tau_p_x.append(tau_mom[0])
        tau_p_y.append(tau_mom[1])
        tau_p_z.append(tau_mom[2])
        tau_p.append(np.linalg.norm(tau_mom))
    data_frame['tau_PX'] = tau_p_x
    data_frame['tau_PY'] = tau_p_y
    data_frame['tau_PZ'] = tau_p_z
    data_frame['tau_P'] = tau_p
    return data_frame


def mass(frame_array):
    return frame_array[0] ** 2 - frame_array[1] ** 2 + frame_array[2] ** 2 + frame_array[3] ** 2


def momentum(frame_array):
    mom = pd.concat([frame_array[1], frame_array[2], + frame_array[3]], axis=1)
    mom.columns = ['X', 'Y', 'Z']
    return mom


def plot_result(data_frame):
    data_frame['muon_not_tau_P'] = data_frame['mu1_P']
    data_frame.loc[data_frame['muon_from_tau'] == 1, 'muon_not_tau_P'] = data_frame['mu2_P']
    data_frame['muon_not_tau_PX'] = data_frame['mu1_PX']
    data_frame.loc[data_frame['muon_from_tau'] == 1, 'muon_not_tau_PX'] = data_frame['mu2_PX']
    data_frame['muon_not_tau_PY'] = data_frame['mu1_PY']
    data_frame.loc[data_frame['muon_from_tau'] == 1, 'muon_not_tau_PY'] = data_frame['mu2_PY']
    data_frame['muon_not_tau_PZ'] = data_frame['mu1_PZ']
    data_frame.loc[data_frame['muon_from_tau'] == 1, 'muon_not_tau_PZ'] = data_frame['mu2_PZ']
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['muon_not_tau_P', 'mu'], ['tau_P', 'tau']]
    particles = ['Kminus_P', 'proton_P', 'muon_not_tau_P', 'tau_P']

    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    plt.hist(sum_m, bins=100, range=[0, 40000])
    plt.show()
    return sum_m


def get_missing_mass(data_frame):
    particles_associations1 = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    particles1 = ['Kminus_P', 'proton_P', 'mu1_P']
    particles_associations2 = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu2_P', 'mu']]
    particles2 = ['Kminus_P', 'proton_P', 'mu2_P']
    lb_energy = np.sqrt(data_frame['Lb_P'] ** 2 + masses['Lb'] ** 2)
    energy = lb_energy - sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations1])
    mom_x = data_frame['Lb_PX'] - sum([data_frame[i + 'X'] for i in particles1])
    mom_y = data_frame['Lb_PY'] - sum([data_frame[i + 'Y'] for i in particles1])
    mom_z = data_frame['Lb_PZ'] - sum([data_frame[i + 'Z'] for i in particles1])

    missing_mass1 = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    print(missing_mass1.describe())
    data_frame['missing_mass1'] = missing_mass1

    energy = lb_energy - sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations2])
    mom_x = data_frame['Lb_PX'] - sum([data_frame[i + 'X'] for i in particles2])
    mom_y = data_frame['Lb_PY'] - sum([data_frame[i + 'Y'] for i in particles2])
    mom_z = data_frame['Lb_PZ'] - sum([data_frame[i + 'Z'] for i in particles2])

    missing_mass2 = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    print(missing_mass2.describe())
    data_frame['missing_mass2'] = missing_mass2
    # plt.hist(missing_mass1, bins=1000)
    # plt.show()
    # plt.hist(missing_mass2, bins=1000)
    # plt.show()
    return data_frame


if __name__ == '__main__':
    a = load_data(add_branches())
    a.dropna(inplace=True)
    df = get_missing_mass(a)
    df, vec = retrieve_vertices(df)
    df = get_missing_mass(df)
    df, t1, t2 = transverse_momentum(df, vec)
    df = line_plane_intersection(df)
    df = tau_momentum_mass(df)
    plot_result(df)
