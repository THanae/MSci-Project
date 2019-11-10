from data_loader import load_data, add_branches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

masses = {'mu': 105.658, 'proton': 938.272, 'K': 493.677, 'pi': 139.57, 'L': 5260, 'tau': 1777}


def retrieve_vertices(data_frame):
    interesting_indices = []
    all_distances = []
    vectors = []
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        pv_xyz = [ts['Lb_PV_X'], ts['Lb_PV_Y'], ts['Lb_PV_Z']]
        end_xyz = [ts['Lb_ENDVERTEX_X'], ts['Lb_ENDVERTEX_Y'], ts['Lb_ENDVERTEX_Z']]
        # errors_pv = [ts['PVXERR'], ts['PVYERR'], ts['PVZERR']]
        # errors_end = [ts['Lb_ENDVERTEX_XERR'], ts['Lb_ENDVERTEX_YERR'], ts['Lb_ENDVERTEX_ZERR']]
        distance = np.linalg.norm(np.array(pv_xyz) - np.array(end_xyz))
        vector = np.array(end_xyz) - np.array(pv_xyz)
        vectors.append(vector)
        all_distances.append(distance)
    data_frame['distances'] = all_distances
    data_frame['vectors'] = vectors
    data_frame = data_frame[data_frame['distances'] > 18]  # should be changed according to what we want
    data_frame = data_frame.drop('distances', axis=1)
    data_frame = data_frame.reset_index()

    return data_frame, vectors


def line_plane_intersection(data_frame):
    intersections = []
    muon_from_tau = []
    for i in range(len(data_frame)):
        end_xyz = [data_frame['Lb_ENDVERTEX_X'][i], data_frame['Lb_ENDVERTEX_Y'][i], data_frame['Lb_ENDVERTEX_Z'][i]]
        momentum_k = [data_frame['Kminus_PX'][i], data_frame['Kminus_PY'][i], data_frame['Kminus_PZ'][i]]
        momentum_p = [data_frame['proton_PX'][i], data_frame['proton_PY'][i], data_frame['proton_PZ'][i]]
        momentum_mu1 = [data_frame['mu1_PX'][i], data_frame['mu1_PY'][i], data_frame['mu1_PZ'][i]]
        momentum_mu2 = [data_frame['mu2_PX'][i], data_frame['mu2_PY'][i], data_frame['mu2_PZ'][i]]
        point_muon_1 = [0, 0, 0]  # need to fill in
        point_muon_2 = [0, 0, 0]  # need to fill in
        # TODO find if intersection between line and plane (make sure line is not in plane but actually intersecting it)
        # define plane with k and p, and check for both muons
        # allow for some error: there will be uncertainties
        intersection = np.array([0, 0, 0])  # need to fill in
        muon_from_tau.append(0)  # 0 for none, 1 for muon 1 and 2 for muon 2
        intersections.append(intersection)
    data_frame['muon_from_tau'] = muon_from_tau
    data_frame['tau_decay_point'] = intersections
    data_frame = data_frame[data_frame['muon_from_tau'] > 0]
    data_frame = data_frame.reset_index()
    return data_frame, data_frame['tau_decay_point']


def transverse_momentum(data_frame, vectors):
    # TODO what are the units
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
        par1 = np.dot(k_vector, par_vector) + np.dot(p_vector, par_vector) + np.dot(m1_vector, par_vector)
        par2 = np.dot(k_vector, par_vector) + np.dot(p_vector, par_vector) + np.dot(m2_vector, par_vector)
        transverse_momentum1 = k_vector + p_vector + m1_vector - par1
        transverse_momentum2 = k_vector + p_vector + m2_vector - par2
        transverse_momenta1.append(transverse_momentum1)
        transverse_momenta2.append(transverse_momentum2)
    data_frame['transverse_momentum1'], data_frame['transverse_momentum2'] = transverse_momenta1, transverse_momenta2
    return data_frame, transverse_momenta1, transverse_momenta2


def tau_momentum_mass(data_frame, transverse_momenta):
    angles, tau_momenta, tau_p, tau_distances_travelled = [], [], [], []
    for i in range(len(data_frame)):
        temp_series = data_frame.loc[i]
        end_xyz = [temp_series['Lb_ENDVERTEX_X'], temp_series['Lb_ENDVERTEX_Y'], temp_series['Lb_ENDVERTEX_Z']]
        tau_vector = temp_series['tau_decay_point'] - end_xyz
        vector = temp_series['vectors']
        tau_distance = np.linalg.norm(tau_vector)
        tau_distances_travelled.append(tau_distance)
        angle = np.arccos(np.dot(tau_vector, vector) / (np.linalg.norm(tau_vector), np.linalg.norm(vector)))
        angles.append(angle)
        if temp_series['muon_from_tau'] == 1:
            tau_mom = temp_series['transverse_momentum2'] / np.tan(angle)
        else:
            tau_mom = temp_series['transverse_momentum1'] / np.tan(angle)
        tau_momenta.append(tau_mom)
        tau_p.append(np.linalg.norm(tau_mom))
    data_frame['tau_mom'] = tau_momenta
    data_frame['tau_P'] = tau_p
    return data_frame, tau_momenta


def mass(frame_array):
    return frame_array[0] ** 2 - frame_array[1] ** 2 + frame_array[2] ** 2 + frame_array[3] ** 2


def momentum(frame_array):
    mom = pd.concat([frame_array[1], frame_array[2], + frame_array[3]], axis=1)
    mom.columns = ['X', 'Y', 'Z']
    return mom


def plot_result(data_frame):
    k_minus = [np.sqrt(data_frame['Kminus_P'] ** 2 + masses['K'] ** 2), data_frame['Kminus_PX'],
               data_frame['Kminus_PY'], data_frame['Kminus_PZ']]
    proton = [np.sqrt(data_frame['proton_P'] ** 2 + masses['proton'] ** 2), data_frame['proton_PX'],
              data_frame['proton_PY'], data_frame['proton_PZ']]
    muon_1 = [np.sqrt(data_frame['mu1_P'] ** 2 + masses['mu'] ** 2), data_frame['mu1_PX'], data_frame['mu1_PY'],
              data_frame['mu1_PZ']]
    muon_2 = [np.sqrt(data_frame['mu2_P'] ** 2 + masses['mu'] ** 2), data_frame['mu2_PX'], data_frame['mu2_PY'],
              data_frame['mu2_PZ']]
    muon_not_tau = [muon_1[i] if data_frame['muon_from_tau'][i] == 1 else muon_2[i] for i in range(len(data_frame))]
    tau = [np.sqrt(data_frame['tau_P'] ** 2 + masses['tau'] ** 2), data_frame['tau_mom'][0], data_frame['tau_mom'][1],
           data_frame['tau_mom'][2]]
    sum = np.array(k_minus) + np.array(proton) + np.array(muon_not_tau) + np.array(tau)
    sum.columns = ['P', 'X', 'Y', 'Z']
    momenta_squared = sum ** 2
    sum_m = np.sqrt(momenta_squared['P'] - momenta_squared['X'] - momenta_squared['Y'] - momenta_squared['Z'])
    plt.hist(sum_m, bins=50)
    return sum_m


if __name__ == '__main__':
    a = load_data(add_branches())
    df, vec = retrieve_vertices(a)
    df, t1, t2 = transverse_momentum(df, vec)
