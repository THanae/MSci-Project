from data_loader import load_data, add_branches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

masses = {'mu': 105.658, 'proton': 938.272, 'K': 493.677, 'pi': 139.57, 'Lb': 5260, 'tau': 1777}


def retrieve_vertices(data_frame):
    all_distances, vectors, errors = [], [], []
    for i in range(len(data_frame)):
        ts = data_frame.loc[i]
        pv_xyz = [ts['Lb_OWNPV_X'], ts['Lb_OWNPV_Y'], ts['Lb_OWNPV_Z']]
        end_xyz = [ts['pKmu_ENDVERTEX_X'], ts['pKmu_ENDVERTEX_Y'], ts['pKmu_ENDVERTEX_Z']]
        errors_pv = [ts['Lb_OWNPV_XERR'], ts['Lb_OWNPV_YERR'], ts['Lb_OWNPV_ZERR']]
        errors_end = [ts['pKmu_REFP_COVXX'], ts['pKmu_REFP_COVYY'], ts['pKmu_REFP_COVZZ']]
        print(errors_end, errors_pv)
        distance = np.linalg.norm(np.array(pv_xyz) - np.array(end_xyz))
        vector = np.array(end_xyz) - np.array(pv_xyz)
        error_x = np.sqrt((errors_pv[0]) ** 2 + (errors_end[0]))
        error_y = np.sqrt((errors_pv[1]) ** 2 + (errors_end[1]))
        error_z = np.sqrt((errors_pv[2]) ** 2 + (errors_end[2]))
        errors.append([error_x, error_y, error_z])
        vectors.append(vector)
        all_distances.append(distance)
    data_frame['distances'] = all_distances
    data_frame['vectors'] = vectors
    data_frame['vectors_errors'] = errors
    # data_frame = data_frame[data_frame['distances'] > 18]  # should be changed according to what we want
    # data_frame = data_frame[(data_frame['missing_mass1'] > masses['tau'] - masses['mu'])]
    # data_frame = data_frame[data_frame['missing_mass2'] > masses['tau'] - masses['mu']]
    data_frame = data_frame.drop('distances', axis=1)
    data_frame = data_frame.reset_index()
    return data_frame, vectors


def line_plane_intersection(data_frame):
    intersections, muon_from_tau = [], []
    for i in range(len(data_frame)):
        end_xyz = [data_frame['pKmu_ENDVERTEX_X'][i], data_frame['pKmu_ENDVERTEX_Y'][i],
                   data_frame['pKmu_ENDVERTEX_Z'][i]]
        momentum_pkmu = [data_frame['pKmu_PX'][i], data_frame['pKmu_PY'][i], data_frame['pKmu_PZ'][i]]
        momentum_tauMu = [data_frame['tauMu_PX'][i], data_frame['tauMu_PY'][i], data_frame['tauMu_PZ'][i]]
        point_tau_mu = [data_frame['tauMu_REFPX'][i], data_frame['tauMu_REFPY'][i], data_frame['tauMu_REFPZ'][i]]
        vector_plane_lb = data_frame['vectors'][i]
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
            ordinate = [point_tau_mu[i] - end_xyz[i] for i in range(3)]
            possible_intersection = np.linalg.solve(coefficient_matrix, ordinate)
            possible_calculation_s = 1 / determinant * sum([inverse_A[0][i] * ordinate[i] for i in range(3)])
            possible_calculation_t = 1 / determinant * sum([inverse_A[1][i] * ordinate[i] for i in range(3)])
            possible_calculation_u = 1 / determinant * sum([inverse_A[2][i] * ordinate[i] for i in range(3)])
            print(possible_intersection)
            print([possible_calculation_s, possible_calculation_t, possible_calculation_u])
            possible_intersection = np.array(momentum_tauMu) * possible_calculation_u + np.array(point_tau_mu)
            print(possible_intersection)
            du_dp_k_mu_ref_x = - inverse_A[2][0] / determinant
            du_dp_k_mu_ref_y = - inverse_A[2][1] / determinant
            du_dp_k_mu_ref_z = - inverse_A[2][2] / determinant
            du_d_tau_ref_x = inverse_A[2][0] / determinant
            du_d_tau_ref_y = inverse_A[2][1] / determinant
            du_d_tau_ref_z = inverse_A[2][2] / determinant
            du_d_vector_lb_x = ((vector_with_mu1[1] * ordinate[2] - vector_with_mu1[2] * ordinate[1]) - (
                    (momentum_tauMu[1] * vector_with_mu1[2] - momentum_tauMu[2] * vector_with_mu1[1]
                     ) * possible_calculation_u)) / determinant
            du_d_vector_lb_y = ((vector_with_mu1[2] * ordinate[0] - vector_with_mu1[0] * ordinate[2]) - (
                    (momentum_tauMu[2] * vector_with_mu1[0] - momentum_tauMu[0] * vector_with_mu1[2]
                     ) * possible_calculation_u)) / determinant
            du_d_vector_lb_z = ((vector_with_mu1[0] * ordinate[1] - vector_with_mu1[1] * ordinate[0]) - (
                    (momentum_tauMu[0] * vector_with_mu1[1] - momentum_tauMu[1] * vector_with_mu1[0]
                     ) * possible_calculation_u)) / determinant
            du_dp_k_mu_mom_x = ((vector_plane_lb[2] * ordinate[1] - vector_plane_lb[1] * ordinate[2]) - (
                    (momentum_tauMu[2] * vector_plane_lb[1] - momentum_tauMu[1] * vector_plane_lb[2]
                     ) * possible_calculation_u)) / determinant
            du_dp_k_mu_mom_y = ((vector_plane_lb[0] * ordinate[2] - vector_plane_lb[2] * ordinate[0]) - (
                    (momentum_tauMu[0] * vector_plane_lb[2] - momentum_tauMu[2] * vector_plane_lb[0]
                     ) * possible_calculation_u)) / determinant
            du_dp_k_mu_mom_z = ((vector_plane_lb[1] * ordinate[0] - vector_plane_lb[0] * ordinate[1]) - (
                    (momentum_tauMu[1] * vector_plane_lb[0] - momentum_tauMu[0] * vector_plane_lb[1]
                     ) * possible_calculation_u)) / determinant
            du_d_tau_mom_x = -(vector_with_mu1[1] * vector_plane_lb[2] - vector_plane_lb[1] * vector_with_mu1[
                2]) * possible_calculation_u / determinant
            du_d_tau_mom_y = -(vector_with_mu1[2] * vector_plane_lb[0] - vector_plane_lb[2] * vector_with_mu1[
                0]) * possible_calculation_u / determinant
            du_d_tau_mom_z = -(vector_with_mu1[0] * vector_plane_lb[1] - vector_plane_lb[0] * vector_with_mu1[
                1]) * possible_calculation_u / determinant
            # sigma u is error in u - here for reference for now
            sigma_u = np.sqrt(du_dp_k_mu_ref_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                              du_dp_k_mu_ref_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                              du_dp_k_mu_ref_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                              du_d_tau_ref_x ** 2 * data_frame['tauMu_REFP_COVXX'][i] +
                              du_d_tau_ref_y ** 2 * data_frame['tauMu_REFP_COVYY'][i] +
                              du_d_tau_ref_z ** 2 * data_frame['tauMu_REFP_COVZZ'][i] +
                              du_d_vector_lb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                              du_d_vector_lb_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                              du_d_vector_lb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                              du_d_vector_lb_x ** 2 * data_frame['Lb_OWNPV_XERR'][i] ** 2 +
                              du_d_vector_lb_y ** 2 * data_frame['Lb_OWNPV_YERR'][i] ** 2 +
                              du_d_vector_lb_z ** 2 * data_frame['Lb_OWNPV_ZERR'][i] ** 2 +
                              du_dp_k_mu_mom_x ** 2 * data_frame['pKmu_P_COVXX'][i] +
                              du_dp_k_mu_mom_y ** 2 * data_frame['pKmu_P_COVYY'][i] +
                              du_dp_k_mu_mom_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                              du_d_tau_mom_x ** 2 * data_frame['tauMu_P_COVXX'][i] +
                              du_d_tau_mom_y ** 2 * data_frame['tauMu_P_COVYY'][i] +
                              du_d_tau_mom_z ** 2 * data_frame['tauMu_P_COVZZ'][i] +
                              2 * du_dp_k_mu_mom_x * du_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                              2 * du_dp_k_mu_mom_x * du_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                              2 * du_dp_k_mu_mom_x * du_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                              2 * du_dp_k_mu_mom_y * du_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                              2 * du_dp_k_mu_mom_y * du_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                              2 * du_dp_k_mu_mom_y * du_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                              2 * du_dp_k_mu_mom_z * du_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                              2 * du_dp_k_mu_mom_z * du_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                              2 * du_dp_k_mu_mom_z * du_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                              2 * du_d_tau_mom_x * du_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PX_X'][i] +
                              2 * du_d_tau_mom_x * du_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PX_Y'][i] +
                              2 * du_d_tau_mom_x * du_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PX_Z'][i] +
                              2 * du_d_tau_mom_y * du_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PY_X'][i] +
                              2 * du_d_tau_mom_y * du_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PY_Y'][i] +
                              2 * du_d_tau_mom_y * du_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PY_Z'][i] +
                              2 * du_d_tau_mom_z * du_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PZ_X'][i] +
                              2 * du_d_tau_mom_z * du_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PZ_Y'][i] +
                              2 * du_d_tau_mom_z * du_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PZ_Z'][i] +
                              2 * du_d_vector_lb_x * du_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                              2 * du_d_vector_lb_x * du_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                              2 * du_d_vector_lb_x * du_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                              2 * du_d_vector_lb_y * du_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                              2 * du_d_vector_lb_y * du_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                              2 * du_d_vector_lb_y * du_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                              2 * du_d_vector_lb_z * du_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                              2 * du_d_vector_lb_z * du_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                              2 * du_d_vector_lb_z * du_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                              2 * du_d_vector_lb_x * du_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXX'][i] +
                              2 * du_d_vector_lb_x * du_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVXY'][i] +
                              2 * du_d_vector_lb_x * du_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVXZ'][i] +
                              2 * du_d_vector_lb_y * du_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXY'][i] +
                              2 * du_d_vector_lb_y * du_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYY'][i] +
                              2 * du_d_vector_lb_y * du_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVYZ'][i] +
                              2 * du_d_vector_lb_z * du_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXZ'][i] +
                              2 * du_d_vector_lb_z * du_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYZ'][i] +
                              2 * du_d_vector_lb_z * du_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVZZ'][i])

            # here, error on intersection point - error on ref_x + u*tau_momx, and so on - derivatives will be different
            # possible_intersection = np.array(momentum_tauMu) * possible_calculation_u + np.array(point_tau_mu)
            # derivatives will be the same apart from reef_x, ref_y, ref_z, and also need to multiply accordingly
            # and di_d_tau_mom_x, di_d_tau_mom_y, di_d_tau_mom_z
            # first do the x one
            di_dp_k_mu_ref_x = 1 - inverse_A[2][0] / determinant * momentum_tauMu[0]
            di_dp_k_mu_ref_y = - inverse_A[2][1] / determinant * momentum_tauMu[0]
            di_dp_k_mu_ref_z = - inverse_A[2][2] / determinant * momentum_tauMu[0]
            di_d_tau_ref_x = inverse_A[2][0] / determinant * momentum_tauMu[0]
            di_d_tau_ref_y = inverse_A[2][1] / determinant * momentum_tauMu[0]
            di_d_tau_ref_z = inverse_A[2][2] / determinant * momentum_tauMu[0]
            di_d_vector_lb_x = ((vector_with_mu1[1] * ordinate[2] - vector_with_mu1[2] * ordinate[1]) - (
                    (momentum_tauMu[1] * vector_with_mu1[2] - momentum_tauMu[2] * vector_with_mu1[1]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
            di_d_vector_lb_y = ((vector_with_mu1[2] * ordinate[0] - vector_with_mu1[0] * ordinate[2]) - (
                    (momentum_tauMu[2] * vector_with_mu1[0] - momentum_tauMu[0] * vector_with_mu1[2]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
            di_d_vector_lb_z = ((vector_with_mu1[0] * ordinate[1] - vector_with_mu1[1] * ordinate[0]) - (
                    (momentum_tauMu[0] * vector_with_mu1[1] - momentum_tauMu[1] * vector_with_mu1[0]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
            di_dp_k_mu_mom_x = ((vector_plane_lb[2] * ordinate[1] - vector_plane_lb[1] * ordinate[2]) - (
                    (momentum_tauMu[2] * vector_plane_lb[1] - momentum_tauMu[1] * vector_plane_lb[2]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
            di_dp_k_mu_mom_y = ((vector_plane_lb[0] * ordinate[2] - vector_plane_lb[2] * ordinate[0]) - (
                    (momentum_tauMu[0] * vector_plane_lb[2] - momentum_tauMu[2] * vector_plane_lb[0]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
            di_dp_k_mu_mom_z = ((vector_plane_lb[1] * ordinate[0] - vector_plane_lb[0] * ordinate[1]) - (
                    (momentum_tauMu[1] * vector_plane_lb[0] - momentum_tauMu[0] * vector_plane_lb[1]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
            di_d_tau_mom_x = -(2 * vector_with_mu1[1] * vector_plane_lb[2] - 2 * vector_plane_lb[1] * vector_with_mu1[
                2]) * possible_calculation_u / determinant * momentum_tauMu[0]
            di_d_tau_mom_y = -(vector_with_mu1[2] * vector_plane_lb[0] - vector_plane_lb[2] * vector_with_mu1[
                0]) * possible_calculation_u / determinant * momentum_tauMu[0]
            di_d_tau_mom_z = -(vector_with_mu1[0] * vector_plane_lb[1] - vector_plane_lb[0] * vector_with_mu1[
                1]) * possible_calculation_u / determinant * momentum_tauMu[0]
            sigma_i_x = np.sqrt(di_dp_k_mu_ref_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                                di_dp_k_mu_ref_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                                di_dp_k_mu_ref_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                                di_d_tau_ref_x ** 2 * data_frame['tauMu_REFP_COVXX'][i] +
                                di_d_tau_ref_y ** 2 * data_frame['tauMu_REFP_COVYY'][i] +
                                di_d_tau_ref_z ** 2 * data_frame['tauMu_REFP_COVZZ'][i] +
                                di_d_vector_lb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                                di_d_vector_lb_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                                di_d_vector_lb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                                di_d_vector_lb_x ** 2 * data_frame['Lb_OWNPV_XERR'][i] ** 2 +
                                di_d_vector_lb_y ** 2 * data_frame['Lb_OWNPV_YERR'][i] ** 2 +
                                di_d_vector_lb_z ** 2 * data_frame['Lb_OWNPV_ZERR'][i] ** 2 +
                                di_dp_k_mu_mom_x ** 2 * data_frame['pKmu_P_COVXX'][i] +
                                di_dp_k_mu_mom_y ** 2 * data_frame['pKmu_P_COVYY'][i] +
                                di_dp_k_mu_mom_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                                di_d_tau_mom_x ** 2 * data_frame['tauMu_P_COVXX'][i] +
                                di_d_tau_mom_y ** 2 * data_frame['tauMu_P_COVYY'][i] +
                                di_d_tau_mom_z ** 2 * data_frame['tauMu_P_COVZZ'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PX_X'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PX_Y'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PX_Z'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PY_X'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PY_Y'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PY_Z'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PZ_X'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXX'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVXY'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVXZ'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXY'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYY'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVYZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVZZ'][i])

            # then the y one
            di_dp_k_mu_ref_x = - inverse_A[2][0] / determinant * momentum_tauMu[1]
            di_dp_k_mu_ref_y = 1 - inverse_A[2][1] / determinant * momentum_tauMu[1]
            di_dp_k_mu_ref_z = - inverse_A[2][2] / determinant * momentum_tauMu[1]
            di_d_tau_ref_x = inverse_A[2][0] / determinant * momentum_tauMu[1]
            di_d_tau_ref_y = inverse_A[2][1] / determinant * momentum_tauMu[1]
            di_d_tau_ref_z = inverse_A[2][2] / determinant * momentum_tauMu[1]
            di_d_vector_lb_x = ((vector_with_mu1[1] * ordinate[2] - vector_with_mu1[2] * ordinate[1]) - (
                    (momentum_tauMu[1] * vector_with_mu1[2] - momentum_tauMu[2] * vector_with_mu1[1]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
            di_d_vector_lb_y = ((vector_with_mu1[2] * ordinate[0] - vector_with_mu1[0] * ordinate[2]) - (
                    (momentum_tauMu[2] * vector_with_mu1[0] - momentum_tauMu[0] * vector_with_mu1[2]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
            di_d_vector_lb_z = ((vector_with_mu1[0] * ordinate[1] - vector_with_mu1[1] * ordinate[0]) - (
                    (momentum_tauMu[0] * vector_with_mu1[1] - momentum_tauMu[1] * vector_with_mu1[0]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
            di_dp_k_mu_mom_x = ((vector_plane_lb[2] * ordinate[1] - vector_plane_lb[1] * ordinate[2]) - (
                    (momentum_tauMu[2] * vector_plane_lb[1] - momentum_tauMu[1] * vector_plane_lb[2]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
            di_dp_k_mu_mom_y = ((vector_plane_lb[0] * ordinate[2] - vector_plane_lb[2] * ordinate[0]) - (
                    (momentum_tauMu[0] * vector_plane_lb[2] - momentum_tauMu[2] * vector_plane_lb[0]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
            di_dp_k_mu_mom_z = ((vector_plane_lb[1] * ordinate[0] - vector_plane_lb[0] * ordinate[1]) - (
                    (momentum_tauMu[1] * vector_plane_lb[0] - momentum_tauMu[0] * vector_plane_lb[1]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
            di_d_tau_mom_x = -(vector_with_mu1[1] * vector_plane_lb[2] - vector_plane_lb[1] * vector_with_mu1[
                2]) * possible_calculation_u / determinant * momentum_tauMu[1]
            di_d_tau_mom_y = -(2 * vector_with_mu1[2] * vector_plane_lb[0] - 2 * vector_plane_lb[2] * vector_with_mu1[
                0]) * possible_calculation_u / determinant * momentum_tauMu[1]
            di_d_tau_mom_z = -(vector_with_mu1[0] * vector_plane_lb[1] - vector_plane_lb[0] * vector_with_mu1[
                1]) * possible_calculation_u / determinant * momentum_tauMu[1]
            sigma_i_y = np.sqrt(di_dp_k_mu_ref_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                                di_dp_k_mu_ref_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                                di_dp_k_mu_ref_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                                di_d_tau_ref_x ** 2 * data_frame['tauMu_REFP_COVXX'][i] +
                                di_d_tau_ref_y ** 2 * data_frame['tauMu_REFP_COVYY'][i] +
                                di_d_tau_ref_z ** 2 * data_frame['tauMu_REFP_COVZZ'][i] +
                                di_d_vector_lb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                                di_d_vector_lb_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                                di_d_vector_lb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                                di_d_vector_lb_x ** 2 * data_frame['Lb_OWNPV_XERR'][i] ** 2 +
                                di_d_vector_lb_y ** 2 * data_frame['Lb_OWNPV_YERR'][i] ** 2 +
                                di_d_vector_lb_z ** 2 * data_frame['Lb_OWNPV_ZERR'][i] ** 2 +
                                di_dp_k_mu_mom_x ** 2 * data_frame['pKmu_P_COVXX'][i] +
                                di_dp_k_mu_mom_y ** 2 * data_frame['pKmu_P_COVYY'][i] +
                                di_dp_k_mu_mom_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                                di_d_tau_mom_x ** 2 * data_frame['tauMu_P_COVXX'][i] +
                                di_d_tau_mom_y ** 2 * data_frame['tauMu_P_COVYY'][i] +
                                di_d_tau_mom_z ** 2 * data_frame['tauMu_P_COVZZ'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PX_X'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PX_Y'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PX_Z'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PY_X'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PY_Y'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PY_Z'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PZ_X'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXX'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVXY'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVXZ'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXY'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYY'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVYZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVZZ'][i])
            # then the z one
            di_dp_k_mu_ref_x = - inverse_A[2][0] / determinant * momentum_tauMu[2]
            di_dp_k_mu_ref_y = - inverse_A[2][1] / determinant * momentum_tauMu[2]
            di_dp_k_mu_ref_z = 1 - inverse_A[2][2] / determinant * momentum_tauMu[2]
            di_d_tau_ref_x = inverse_A[2][0] / determinant * momentum_tauMu[2]
            di_d_tau_ref_y = inverse_A[2][1] / determinant * momentum_tauMu[2]
            di_d_tau_ref_z = inverse_A[2][2] / determinant * momentum_tauMu[2]
            di_d_vector_lb_x = ((vector_with_mu1[1] * ordinate[2] - vector_with_mu1[2] * ordinate[1]) - (
                    (momentum_tauMu[1] * vector_with_mu1[2] - momentum_tauMu[2] * vector_with_mu1[1]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
            di_d_vector_lb_y = ((vector_with_mu1[2] * ordinate[0] - vector_with_mu1[0] * ordinate[2]) - (
                    (momentum_tauMu[2] * vector_with_mu1[0] - momentum_tauMu[0] * vector_with_mu1[2]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
            di_d_vector_lb_z = ((vector_with_mu1[0] * ordinate[1] - vector_with_mu1[1] * ordinate[0]) - (
                    (momentum_tauMu[0] * vector_with_mu1[1] - momentum_tauMu[1] * vector_with_mu1[0]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
            di_dp_k_mu_mom_x = ((vector_plane_lb[2] * ordinate[1] - vector_plane_lb[1] * ordinate[2]) - (
                    (momentum_tauMu[2] * vector_plane_lb[1] - momentum_tauMu[1] * vector_plane_lb[2]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
            di_dp_k_mu_mom_y = ((vector_plane_lb[0] * ordinate[2] - vector_plane_lb[2] * ordinate[0]) - (
                    (momentum_tauMu[0] * vector_plane_lb[2] - momentum_tauMu[2] * vector_plane_lb[0]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
            di_dp_k_mu_mom_z = ((vector_plane_lb[1] * ordinate[0] - vector_plane_lb[0] * ordinate[1]) - (
                    (momentum_tauMu[1] * vector_plane_lb[0] - momentum_tauMu[0] * vector_plane_lb[1]
                     ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
            di_d_tau_mom_x = -(vector_with_mu1[1] * vector_plane_lb[2] - vector_plane_lb[1] * vector_with_mu1[
                2]) * possible_calculation_u / determinant * momentum_tauMu[2]
            di_d_tau_mom_y = -(vector_with_mu1[2] * vector_plane_lb[0] - vector_plane_lb[2] * vector_with_mu1[
                0]) * possible_calculation_u / determinant * momentum_tauMu[2]
            di_d_tau_mom_z = -(2 * vector_with_mu1[0] * vector_plane_lb[1] - 2 * vector_plane_lb[0] * vector_with_mu1[
                1]) * possible_calculation_u / determinant * momentum_tauMu[2]
            sigma_i_z = np.sqrt(di_dp_k_mu_ref_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                                di_dp_k_mu_ref_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                                di_dp_k_mu_ref_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                                di_d_tau_ref_x ** 2 * data_frame['tauMu_REFP_COVXX'][i] +
                                di_d_tau_ref_y ** 2 * data_frame['tauMu_REFP_COVYY'][i] +
                                di_d_tau_ref_z ** 2 * data_frame['tauMu_REFP_COVZZ'][i] +
                                di_d_vector_lb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                                di_d_vector_lb_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                                di_d_vector_lb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                                di_d_vector_lb_x ** 2 * data_frame['Lb_OWNPV_XERR'][i] ** 2 +
                                di_d_vector_lb_y ** 2 * data_frame['Lb_OWNPV_YERR'][i] ** 2 +
                                di_d_vector_lb_z ** 2 * data_frame['Lb_OWNPV_ZERR'][i] ** 2 +
                                di_dp_k_mu_mom_x ** 2 * data_frame['pKmu_P_COVXX'][i] +
                                di_dp_k_mu_mom_y ** 2 * data_frame['pKmu_P_COVYY'][i] +
                                di_dp_k_mu_mom_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                                di_d_tau_mom_x ** 2 * data_frame['tauMu_P_COVXX'][i] +
                                di_d_tau_mom_y ** 2 * data_frame['tauMu_P_COVYY'][i] +
                                di_d_tau_mom_z ** 2 * data_frame['tauMu_P_COVZZ'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                                2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                                2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PX_X'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PX_Y'][i] +
                                2 * di_d_tau_mom_x * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PX_Z'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PY_X'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PY_Y'][i] +
                                2 * di_d_tau_mom_y * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PY_Z'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PZ_X'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_d_tau_mom_z * di_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXX'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVXY'][i] +
                                2 * di_d_vector_lb_x * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVXZ'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXY'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYY'][i] +
                                2 * di_d_vector_lb_y * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVYZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYZ'][i] +
                                2 * di_d_vector_lb_z * di_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVZZ'][i])
            # TODO manage to obtain nicer format
        else:
            possible_intersection = False
        if possible_intersection is not False:
            intersection = possible_intersection
            muon_from_tau.append(2)  # 0 for none, 1 for muon 1 and 2 for muon 2
        # elif possible_intersection2 is not False:
        #     intersection = possible_intersection2
        #     muon_from_tau.append(2)  # 0 for none, 1 for muon 1 and 2 for muon 2
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
    pkmu = [np.sqrt(data_frame['pKmu_P'] ** 2), data_frame['pKmu_PX'], data_frame['pKmu_PY'], data_frame['pKmu_PZ']]
    transverse_momenta = []
    k_momentum, p_momentum = momentum(k_minus), momentum(proton)
    pkmu_momentum = momentum(pkmu)
    m1_momentum = momentum(muon_1)
    for i in range(len(data_frame)):
        par_vector = vectors[i] / np.linalg.norm(vectors[i])
        # k_vector, p_vector = np.array(k_momentum.loc[i]), np.array(p_momentum.loc[i])
        # m1_vector = np.array(m1_momentum.loc[i])
        pkmu_vector = np.array(pkmu_momentum.loc[i])
        # par_k = np.dot(k_vector, par_vector) * k_vector / np.linalg.norm(k_vector)
        # par_p = np.dot(p_vector, par_vector) * p_vector / np.linalg.norm(p_vector)
        # par_kp = par_k + par_p
        # par = par_kp + np.dot(m1_vector, par_vector) * m1_vector / np.linalg.norm(m1_vector)
        # transverse_mom = k_vector + p_vector + m1_vector - par
        par = np.dot(pkmu_vector, par_vector) * pkmu_vector / np.linalg.norm(pkmu_vector)
        transverse_mom = pkmu_vector - par

        def derivative_pkmu(given_vector, number):
            der = - vectors[i][number] / np.linalg.norm(vectors[i]) * (
                    2 * given_vector[0] * np.linalg.norm(given_vector) - given_vector[
                0] ** 2 / np.linalg.norm(given_vector)) / np.linalg.norm(given_vector) ** 2
            return der

        def derivative_vec(given_vector, number):
            der = - pkmu_vector[number] ** 2 / np.linalg.norm(pkmu_vector) * (
                    np.linalg.norm(given_vector) - given_vector[number] ** 2 / np.linalg.norm(
                given_vector)) / np.linalg.norm(given_vector)
            return der

        dt_x_dpkmu_x = 1 + derivative_pkmu(pkmu_vector, 0)
        dt_x_dpkmu_y = derivative_pkmu(pkmu_vector, 1)
        dt_x_dpkmu_z = derivative_pkmu(pkmu_vector, 2)
        dt_x_dlb_x = derivative_vec(vectors[i], 0)
        dt_x_dlb_y = derivative_vec(vectors[i], 1)
        dt_x_dlb_z = derivative_vec(vectors[i], 2)

        dt_y_dpkmu_x = derivative_pkmu(pkmu_vector, 0)
        dt_y_dpkmu_y = 1 + derivative_pkmu(pkmu_vector, 1)
        dt_y_dpkmu_z = derivative_pkmu(pkmu_vector, 2)
        dt_y_dlb_x = derivative_vec(vectors[i], 0)
        dt_y_dlb_y = derivative_vec(vectors[i], 1)
        dt_y_dlb_z = derivative_vec(vectors[i], 2)

        dt_z_dpkmu_x = derivative_pkmu(pkmu_vector, 0)
        dt_z_dpkmu_y = derivative_pkmu(pkmu_vector, 1)
        dt_z_dpkmu_z = 1 + derivative_pkmu(pkmu_vector, 2)
        dt_z_dlb_x = derivative_vec(vectors[i], 0)
        dt_z_dlb_y = derivative_vec(vectors[i], 1)
        dt_z_dlb_z = derivative_vec(vectors[i], 2)
        # error not needed now but here for information
        t_mom_err = [[dt_x_dpkmu_x ** 2 * data_frame['pKmu_P_COVXX'][i] + dt_x_dpkmu_y ** 2 *
                      data_frame['pKmu_P_COVYY'][i] + dt_x_dpkmu_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                      dt_x_dlb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] + dt_x_dlb_y ** 2 *
                      data_frame['pKmu_REFP_COVYY'][i] + dt_x_dlb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                      2 * dt_x_dpkmu_x * dt_x_dlb_x * data_frame['pKmu_P_REFP_COV_PX_X'][
                          i] + 2 * dt_x_dpkmu_x * dt_x_dlb_y * data_frame['pKmu_P_REFP_COV_PX_Y'][
                          i] + 2 * dt_x_dpkmu_x * dt_x_dlb_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                      2 * dt_x_dpkmu_y * dt_x_dlb_x * data_frame['pKmu_P_REFP_COV_PY_X'][
                          i] + 2 * dt_x_dpkmu_y * dt_x_dlb_y * data_frame['pKmu_P_REFP_COV_PY_Y'][
                          i] + 2 * dt_x_dpkmu_y * dt_x_dlb_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                      2 * dt_x_dpkmu_z * dt_x_dlb_x * data_frame['pKmu_P_REFP_COV_PZ_X'][
                          i] + 2 * dt_x_dpkmu_z * dt_x_dlb_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][
                          i] + 2 * dt_x_dpkmu_z * dt_x_dlb_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i]],
                     [dt_y_dpkmu_x ** 2 * data_frame['pKmu_P_COVXX'][i] + dt_y_dpkmu_y ** 2 *
                      data_frame['pKmu_P_COVYY'][i] + dt_y_dpkmu_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                      dt_y_dlb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] + dt_y_dlb_y ** 2 *
                      data_frame['pKmu_REFP_COVYY'][i] + dt_y_dlb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                      2 * dt_y_dpkmu_x * dt_y_dlb_x * data_frame['pKmu_P_REFP_COV_PX_X'][
                          i] + 2 * dt_y_dpkmu_x * dt_y_dlb_y * data_frame['pKmu_P_REFP_COV_PX_Y'][
                          i] + 2 * dt_y_dpkmu_x * dt_y_dlb_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                      2 * dt_y_dpkmu_y * dt_y_dlb_x * data_frame['pKmu_P_REFP_COV_PY_X'][
                          i] + 2 * dt_y_dpkmu_y * dt_y_dlb_y * data_frame['pKmu_P_REFP_COV_PY_Y'][
                          i] + 2 * dt_y_dpkmu_y * dt_y_dlb_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                      2 * dt_y_dpkmu_z * dt_y_dlb_x * data_frame['pKmu_P_REFP_COV_PZ_X'][
                          i] + 2 * dt_y_dpkmu_z * dt_y_dlb_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][
                          i] + 2 * dt_y_dpkmu_z * dt_y_dlb_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i]],
                     [dt_z_dpkmu_x ** 2 * data_frame['pKmu_P_COVXX'][i] + dt_z_dpkmu_y ** 2 *
                      data_frame['pKmu_P_COVYY'][i] + dt_z_dpkmu_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                      dt_z_dlb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] + dt_z_dlb_y ** 2 *
                      data_frame['pKmu_REFP_COVYY'][i] + dt_z_dlb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                      2 * dt_z_dpkmu_x * dt_z_dlb_x * data_frame['pKmu_P_REFP_COV_PX_X'][
                          i] + 2 * dt_z_dpkmu_x * dt_z_dlb_y * data_frame['pKmu_P_REFP_COV_PX_Y'][
                          i] + 2 * dt_z_dpkmu_x * dt_z_dlb_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                      2 * dt_z_dpkmu_y * dt_z_dlb_x * data_frame['pKmu_P_REFP_COV_PY_X'][
                          i] + 2 * dt_z_dpkmu_y * dt_z_dlb_y * data_frame['pKmu_P_REFP_COV_PY_Y'][
                          i] + 2 * dt_z_dpkmu_y * dt_z_dlb_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                      2 * dt_z_dpkmu_z * dt_z_dlb_x * data_frame['pKmu_P_REFP_COV_PZ_X'][
                          i] + 2 * dt_z_dpkmu_z * dt_z_dlb_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][
                          i] + 2 * dt_z_dpkmu_z * dt_z_dlb_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i]]]

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
        momentum_tauMu = [data_frame['tauMu_PX'][i], data_frame['tauMu_PY'][i], data_frame['tauMu_PZ'][i]]
        pkmu_vector = [data_frame['pKmu_PX'][i], data_frame['pKmu_PY'][i], data_frame['pKmu_PZ'][i]]
        vector_plane_lb = data_frame['vectors'][i]
        vector_with_mu1 = [data_frame['pKmu_PX'][i], data_frame['pKmu_PY'][i], data_frame['pKmu_PZ'][i]]
        point_tau_mu = [data_frame['tauMu_REFPX'][i], data_frame['tauMu_REFPY'][i], data_frame['tauMu_REFPZ'][i]]
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
        coefficient_matrix = [[vector_plane_lb[i], vector_with_mu1[i], - momentum_tauMu[i]] for i in range(3)]
        ordinate = [point_tau_mu[i] - end_xyz[i] for i in range(3)]
        possible_calculation_u = 1 / determinant * sum([inverse_A[2][i] * ordinate[i] for i in range(3)])
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
        #####
        di_x_dp_k_mu_ref_x = 1 - inverse_A[2][0] / determinant * momentum_tauMu[0]
        di_x_dp_k_mu_ref_y = - inverse_A[2][1] / determinant * momentum_tauMu[0]
        di_x_dp_k_mu_ref_z = - inverse_A[2][2] / determinant * momentum_tauMu[0]
        di_y_dp_k_mu_ref_x = - inverse_A[2][0] / determinant * momentum_tauMu[1]
        di_y_dp_k_mu_ref_y = 1 - inverse_A[2][1] / determinant * momentum_tauMu[1]
        di_y_dp_k_mu_ref_z = - inverse_A[2][2] / determinant * momentum_tauMu[1]
        di_z_dp_k_mu_ref_x = - inverse_A[2][0] / determinant * momentum_tauMu[2]
        di_z_dp_k_mu_ref_y = - inverse_A[2][1] / determinant * momentum_tauMu[2]
        di_z_dp_k_mu_ref_z = 1 - inverse_A[2][2] / determinant * momentum_tauMu[2]
        dt_x_dp_k_mu_ref_x = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * (((di_x_dp_k_mu_ref_x - 1) * vector_plane_lb[
            0] + di_y_dp_k_mu_ref_x * vector_plane_lb[1] + di_z_dp_k_mu_ref_x * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * (-1 + di_x_dp_k_mu_ref_x) + tau_vector[1] * di_y_dp_k_mu_ref_x +
                                        tau_vector[2] * di_z_dp_k_mu_ref_x) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_x_dp_k_mu_ref_y = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_dp_k_mu_ref_y * vector_plane_lb[
            0] + (di_y_dp_k_mu_ref_y - 1) * vector_plane_lb[1] + di_z_dp_k_mu_ref_y * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_ref_y + tau_vector[1] * (di_y_dp_k_mu_ref_y - 1) +
                                        tau_vector[2] * di_z_dp_k_mu_ref_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_x_dp_k_mu_ref_z = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_dp_k_mu_ref_z * vector_plane_lb[
            0] + di_y_dp_k_mu_ref_z * vector_plane_lb[1] + (di_z_dp_k_mu_ref_z - 1) * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_ref_z + tau_vector[1] * di_y_dp_k_mu_ref_z +
                                        tau_vector[2] * (di_z_dp_k_mu_ref_z - 1)) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        #####
        di_x_d_tau_ref_x = inverse_A[2][0] / determinant * momentum_tauMu[0]
        di_x_d_tau_ref_y = inverse_A[2][1] / determinant * momentum_tauMu[0]
        di_x_d_tau_ref_z = inverse_A[2][2] / determinant * momentum_tauMu[0]
        di_y_d_tau_ref_x = inverse_A[2][0] / determinant * momentum_tauMu[1]
        di_y_d_tau_ref_y = inverse_A[2][1] / determinant * momentum_tauMu[1]
        di_y_d_tau_ref_z = inverse_A[2][2] / determinant * momentum_tauMu[1]
        di_z_d_tau_ref_x = inverse_A[2][0] / determinant * momentum_tauMu[2]
        di_z_d_tau_ref_y = inverse_A[2][1] / determinant * momentum_tauMu[2]
        di_z_d_tau_ref_z = inverse_A[2][2] / determinant * momentum_tauMu[2]
        #####

        dt_x_d_tau_ref_x = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_x * vector_plane_lb[
            0] + di_y_d_tau_ref_x * vector_plane_lb[1] + di_z_d_tau_ref_x * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_x + tau_vector[1] * di_y_d_tau_ref_x +
                                        tau_vector[2] * di_z_d_tau_ref_x) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_x_d_tau_ref_y = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_y * vector_plane_lb[
            0] + di_y_d_tau_ref_y * vector_plane_lb[1] + di_z_d_tau_ref_y * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_y + tau_vector[1] * di_y_d_tau_ref_y +
                                        tau_vector[2] * di_z_d_tau_ref_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_x_d_tau_ref_z = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_z * vector_plane_lb[
            0] + di_y_d_tau_ref_z * vector_plane_lb[1] + di_z_d_tau_ref_z * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_z + tau_vector[1] * di_y_d_tau_ref_z +
                                        tau_vector[2] * di_z_d_tau_ref_z) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2

        par_vector = data_frame['vectors'][i] / np.linalg.norm(data_frame['vectors'][i])
        par = np.dot(pkmu_vector, par_vector) * np.array(pkmu_vector) / np.linalg.norm(pkmu_vector)
        ###
        di_x_dp_k_mu_mom_x = ((vector_plane_lb[2] * ordinate[1] - vector_plane_lb[1] * ordinate[2]) - (
                (momentum_tauMu[2] * vector_plane_lb[1] - momentum_tauMu[1] * vector_plane_lb[2]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
        di_x_dp_k_mu_mom_y = ((vector_plane_lb[0] * ordinate[2] - vector_plane_lb[2] * ordinate[0]) - (
                (momentum_tauMu[0] * vector_plane_lb[2] - momentum_tauMu[2] * vector_plane_lb[0]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
        di_x_dp_k_mu_mom_z = ((vector_plane_lb[1] * ordinate[0] - vector_plane_lb[0] * ordinate[1]) - (
                (momentum_tauMu[1] * vector_plane_lb[0] - momentum_tauMu[0] * vector_plane_lb[1]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
        di_y_dp_k_mu_mom_x = ((vector_plane_lb[2] * ordinate[1] - vector_plane_lb[1] * ordinate[2]) - (
                (momentum_tauMu[2] * vector_plane_lb[1] - momentum_tauMu[1] * vector_plane_lb[2]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
        di_y_dp_k_mu_mom_y = ((vector_plane_lb[0] * ordinate[2] - vector_plane_lb[2] * ordinate[0]) - (
                (momentum_tauMu[0] * vector_plane_lb[2] - momentum_tauMu[2] * vector_plane_lb[0]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
        di_y_dp_k_mu_mom_z = ((vector_plane_lb[1] * ordinate[0] - vector_plane_lb[0] * ordinate[1]) - (
                (momentum_tauMu[1] * vector_plane_lb[0] - momentum_tauMu[0] * vector_plane_lb[1]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
        di_z_dp_k_mu_mom_x = ((vector_plane_lb[2] * ordinate[1] - vector_plane_lb[1] * ordinate[2]) - (
                (momentum_tauMu[2] * vector_plane_lb[1] - momentum_tauMu[1] * vector_plane_lb[2]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
        di_z_dp_k_mu_mom_y = ((vector_plane_lb[0] * ordinate[2] - vector_plane_lb[2] * ordinate[0]) - (
                (momentum_tauMu[0] * vector_plane_lb[2] - momentum_tauMu[2] * vector_plane_lb[0]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
        di_z_dp_k_mu_mom_z = ((vector_plane_lb[1] * ordinate[0] - vector_plane_lb[0] * ordinate[1]) - (
                (momentum_tauMu[1] * vector_plane_lb[0] - momentum_tauMu[0] * vector_plane_lb[1]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
        ###
        dt_x_dp_k_mu_mom_x = (unit_l[0] / np.tan(angle)) * (2 * pkmu_vector[0] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[0] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[0] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[0] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[0] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_x +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_x +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_x) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_x + tau_vector[1] * di_y_dp_k_mu_mom_x + tau_vector[
            2] * di_z_dp_k_mu_mom_x) / np.linalg.norm(tau_vector)) / bottom_content ** 2 + \
                             1 - \
                             ((2 * pkmu_vector[0] * vector_plane_lb[0] + pkmu_vector[1] * vector_plane_lb[1] +
                               pkmu_vector[2] * vector_plane_lb[2]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[0] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[0] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2

        dt_x_dp_k_mu_mom_y = (unit_l[0] / np.tan(angle)) * (2 * pkmu_vector[1] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[1] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[1] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[1] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[0] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_y +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_y +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_y) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_y + tau_vector[1] * di_y_dp_k_mu_mom_y + tau_vector[
            2] * di_z_dp_k_mu_mom_y) / np.linalg.norm(tau_vector)) / bottom_content ** 2 - \
                             ((pkmu_vector[0] * vector_plane_lb[1]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[0] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[1] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2
        dt_x_dp_k_mu_mom_z = (unit_l[0] / np.tan(angle)) * (2 * pkmu_vector[2] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[2] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[2] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[2] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[0] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_z +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_z +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_z) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_z + tau_vector[1] * di_y_dp_k_mu_mom_z + tau_vector[
            2] * di_z_dp_k_mu_mom_z) / np.linalg.norm(tau_vector)) / bottom_content ** 2 - \
                             ((pkmu_vector[0] * vector_plane_lb[2]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[0] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[2] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2
        ###
        di_x_d_vector_lb_x = ((vector_with_mu1[1] * ordinate[2] - vector_with_mu1[2] * ordinate[1]) - (
                (momentum_tauMu[1] * vector_with_mu1[2] - momentum_tauMu[2] * vector_with_mu1[1]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
        di_x_d_vector_lb_y = ((vector_with_mu1[2] * ordinate[0] - vector_with_mu1[0] * ordinate[2]) - (
                (momentum_tauMu[2] * vector_with_mu1[0] - momentum_tauMu[0] * vector_with_mu1[2]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
        di_x_d_vector_lb_z = ((vector_with_mu1[0] * ordinate[1] - vector_with_mu1[1] * ordinate[0]) - (
                (momentum_tauMu[0] * vector_with_mu1[1] - momentum_tauMu[1] * vector_with_mu1[0]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[0]
        di_y_d_vector_lb_x = ((vector_with_mu1[1] * ordinate[2] - vector_with_mu1[2] * ordinate[1]) - (
                (momentum_tauMu[1] * vector_with_mu1[2] - momentum_tauMu[2] * vector_with_mu1[1]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
        di_y_d_vector_lb_y = ((vector_with_mu1[2] * ordinate[0] - vector_with_mu1[0] * ordinate[2]) - (
                (momentum_tauMu[2] * vector_with_mu1[0] - momentum_tauMu[0] * vector_with_mu1[2]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
        di_y_d_vector_lb_z = ((vector_with_mu1[0] * ordinate[1] - vector_with_mu1[1] * ordinate[0]) - (
                (momentum_tauMu[0] * vector_with_mu1[1] - momentum_tauMu[1] * vector_with_mu1[0]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[1]
        di_z_d_vector_lb_x = ((vector_with_mu1[1] * ordinate[2] - vector_with_mu1[2] * ordinate[1]) - (
                (momentum_tauMu[1] * vector_with_mu1[2] - momentum_tauMu[2] * vector_with_mu1[1]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
        di_z_d_vector_lb_y = ((vector_with_mu1[2] * ordinate[0] - vector_with_mu1[0] * ordinate[2]) - (
                (momentum_tauMu[2] * vector_with_mu1[0] - momentum_tauMu[0] * vector_with_mu1[2]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
        di_z_d_vector_lb_z = ((vector_with_mu1[0] * ordinate[1] - vector_with_mu1[1] * ordinate[0]) - (
                (momentum_tauMu[0] * vector_with_mu1[1] - momentum_tauMu[1] * vector_with_mu1[0]
                 ) * possible_calculation_u)) / determinant * momentum_tauMu[2]
        ###
        dt_x_d_vector_lb_x = p_transverse / np.tan(angle) / np.linalg.norm(vector_plane_lb) + \
                             p_transverse / np.tan(angle) * vector_plane_lb[0] * (
                                     -vector_plane_lb[0] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[0]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (tau_vector[0] + vector_plane_lb[0] * di_x_d_vector_lb_x + vector_plane_lb[
                                         1] * di_y_d_vector_lb_x + vector_plane_lb[
                                          2] * di_z_d_vector_lb_x) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[0] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_x + tau_vector[
                                                 1] * di_y_d_vector_lb_x + tau_vector[
                                                         2] * di_z_d_vector_lb_x) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[0] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[0] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[0] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[0] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[0] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[0] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[0] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[0] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2

        dt_x_d_vector_lb_y = p_transverse / np.tan(angle) * vector_plane_lb[0] * (
                -vector_plane_lb[1] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[0]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (vector_plane_lb[0] * di_x_d_vector_lb_y + tau_vector[1] + vector_plane_lb[
                                         1] * di_y_d_vector_lb_y + vector_plane_lb[
                                          2] * di_z_d_vector_lb_y) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[1] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_y + tau_vector[
                                                 1] * di_y_d_vector_lb_y + tau_vector[
                                                         2] * di_z_d_vector_lb_y) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[0] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[1] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[1] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[1] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[1] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[0] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[1] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[1] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2
        dt_x_d_vector_lb_z = p_transverse / np.tan(angle) * vector_plane_lb[0] * (
                -vector_plane_lb[2] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[0]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (vector_plane_lb[0] * di_x_d_vector_lb_z + vector_plane_lb[
                                         1] * di_y_d_vector_lb_z + tau_vector[2] + vector_plane_lb[
                                          2] * di_z_d_vector_lb_z) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[2] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_z + tau_vector[
                                                 1] * di_y_d_vector_lb_z + tau_vector[
                                                         2] * di_z_d_vector_lb_z) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[0] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[2] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[2] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[2] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[2] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[0] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[2] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[2] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2
        ###
        di_x_d_tau_mom_x = -(2 * vector_with_mu1[1] * vector_plane_lb[2] - 2 * vector_plane_lb[1] * vector_with_mu1[
            2]) * possible_calculation_u / determinant * momentum_tauMu[0]
        di_x_d_tau_mom_y = -(vector_with_mu1[2] * vector_plane_lb[0] - vector_plane_lb[2] * vector_with_mu1[
            0]) * possible_calculation_u / determinant * momentum_tauMu[0]
        di_x_d_tau_mom_z = -(vector_with_mu1[0] * vector_plane_lb[1] - vector_plane_lb[0] * vector_with_mu1[
            1]) * possible_calculation_u / determinant * momentum_tauMu[0]
        di_y_d_tau_mom_x = -(vector_with_mu1[1] * vector_plane_lb[2] - vector_plane_lb[1] * vector_with_mu1[
            2]) * possible_calculation_u / determinant * momentum_tauMu[1]
        di_y_d_tau_mom_y = -(2 * vector_with_mu1[2] * vector_plane_lb[0] - 2 * vector_plane_lb[2] * vector_with_mu1[
            0]) * possible_calculation_u / determinant * momentum_tauMu[1]
        di_y_d_tau_mom_z = -(vector_with_mu1[0] * vector_plane_lb[1] - vector_plane_lb[0] * vector_with_mu1[
            1]) * possible_calculation_u / determinant * momentum_tauMu[1]
        di_z_d_tau_mom_x = -(vector_with_mu1[1] * vector_plane_lb[2] - vector_plane_lb[1] * vector_with_mu1[
            2]) * possible_calculation_u / determinant * momentum_tauMu[2]
        di_z_d_tau_mom_y = -(vector_with_mu1[2] * vector_plane_lb[0] - vector_plane_lb[2] * vector_with_mu1[
            0]) * possible_calculation_u / determinant * momentum_tauMu[2]
        di_z_d_tau_mom_z = -(2 * vector_with_mu1[0] * vector_plane_lb[1] - 2 * vector_plane_lb[0] * vector_with_mu1[
            1]) * possible_calculation_u / determinant * momentum_tauMu[2]
        ###
        dt_x_d_tau_mom_x = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * (
                                   (di_x_d_tau_mom_x * vector_plane_lb[0] + di_y_d_tau_mom_x * vector_plane_lb[
                                       1] + di_z_d_tau_mom_x *
                                    vector_plane_lb[2]) * bottom_content - np.dot(vector_plane_lb,
                                                                                  tau_vector) * np.linalg.norm(
                               vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_x + tau_vector[1] * di_y_d_tau_mom_x +
                                                   tau_vector[2] * di_z_d_tau_mom_x) / np.linalg.norm(
                               tau_vector)) / bottom_content ** 2
        dt_x_d_tau_mom_y = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_mom_y * vector_plane_lb[
            0] + di_y_d_tau_mom_y * vector_plane_lb[1] + di_z_d_tau_mom_y * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_y + tau_vector[1] * di_y_d_tau_mom_y +
                                        tau_vector[2] * di_z_d_tau_mom_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_x_d_tau_mom_z = p_transverse * unit_l[0] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_mom_z * vector_plane_lb[
            0] + di_y_d_tau_mom_z * vector_plane_lb[1] + di_z_d_tau_mom_z * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_z + tau_vector[1] * di_y_d_tau_mom_z +
                                        tau_vector[2] * di_z_d_tau_mom_z) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        sigma_tau_x = np.sqrt(dt_x_dp_k_mu_ref_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                              dt_x_dp_k_mu_ref_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                              dt_x_dp_k_mu_ref_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                              dt_x_d_tau_ref_x ** 2 * data_frame['tauMu_REFP_COVXX'][i] +
                              dt_x_d_tau_ref_y ** 2 * data_frame['tauMu_REFP_COVYY'][i] +
                              dt_x_d_tau_ref_z ** 2 * data_frame['tauMu_REFP_COVZZ'][i] +
                              dt_x_d_vector_lb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                              dt_x_d_vector_lb_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                              dt_x_d_vector_lb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                              dt_x_d_vector_lb_x ** 2 * data_frame['Lb_OWNPV_XERR'][i] ** 2 +
                              dt_x_d_vector_lb_y ** 2 * data_frame['Lb_OWNPV_YERR'][i] ** 2 +
                              dt_x_d_vector_lb_z ** 2 * data_frame['Lb_OWNPV_ZERR'][i] ** 2 +
                              dt_x_dp_k_mu_mom_x ** 2 * data_frame['pKmu_P_COVXX'][i] +
                              dt_x_dp_k_mu_mom_y ** 2 * data_frame['pKmu_P_COVYY'][i] +
                              dt_x_dp_k_mu_mom_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                              dt_x_d_tau_mom_x ** 2 * data_frame['tauMu_P_COVXX'][i] +
                              dt_x_d_tau_mom_y ** 2 * data_frame['tauMu_P_COVYY'][i] +
                              dt_x_d_tau_mom_z ** 2 * data_frame['tauMu_P_COVZZ'][i] +
                              2 * dt_x_dp_k_mu_mom_x * dt_x_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                              2 * dt_x_dp_k_mu_mom_x * dt_x_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_x_dp_k_mu_mom_x * dt_x_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_x_dp_k_mu_mom_y * dt_x_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                              2 * dt_x_dp_k_mu_mom_y * dt_x_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_x_dp_k_mu_mom_y * dt_x_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_x_dp_k_mu_mom_z * dt_x_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_x_dp_k_mu_mom_z * dt_x_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_x_dp_k_mu_mom_z * dt_x_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_x_d_tau_mom_x * dt_x_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PX_X'][i] +
                              2 * dt_x_d_tau_mom_x * dt_x_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_x_d_tau_mom_x * dt_x_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_x_d_tau_mom_y * dt_x_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PY_X'][i] +
                              2 * dt_x_d_tau_mom_y * dt_x_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_x_d_tau_mom_y * dt_x_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_x_d_tau_mom_z * dt_x_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_x_d_tau_mom_z * dt_x_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_x_d_tau_mom_z * dt_x_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                              2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                              2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXX'][i] +
                              2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVXY'][i] +
                              2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVXZ'][i] +
                              2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXY'][i] +
                              2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYY'][i] +
                              2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVYZ'][i] +
                              2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXZ'][i] +
                              2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYZ'][i] +
                              2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVZZ'][i])


        dt_y_dp_k_mu_ref_x = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * (((di_x_dp_k_mu_ref_x - 1) * vector_plane_lb[
            0] + di_y_dp_k_mu_ref_x * vector_plane_lb[1] + di_z_dp_k_mu_ref_x * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * (-1 + di_x_dp_k_mu_ref_x) + tau_vector[1] * di_y_dp_k_mu_ref_x +
                                        tau_vector[2] * di_z_dp_k_mu_ref_x) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_y_dp_k_mu_ref_y = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_dp_k_mu_ref_y * vector_plane_lb[
            0] + (di_y_dp_k_mu_ref_y - 1) * vector_plane_lb[1] + di_z_dp_k_mu_ref_y * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_ref_y + tau_vector[1] * (di_y_dp_k_mu_ref_y - 1) +
                                        tau_vector[2] * di_z_dp_k_mu_ref_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_y_dp_k_mu_ref_z = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_dp_k_mu_ref_z * vector_plane_lb[
            0] + di_y_dp_k_mu_ref_z * vector_plane_lb[1] + (di_z_dp_k_mu_ref_z - 1) * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_ref_z + tau_vector[1] * di_y_dp_k_mu_ref_z +
                                        tau_vector[2] * (di_z_dp_k_mu_ref_z - 1)) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2

        dt_y_d_tau_ref_x = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_x * vector_plane_lb[
            0] + di_y_d_tau_ref_x * vector_plane_lb[1] + di_z_d_tau_ref_x * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_x + tau_vector[1] * di_y_d_tau_ref_x +
                                        tau_vector[2] * di_z_d_tau_ref_x) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_y_d_tau_ref_y = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_y * vector_plane_lb[
            0] + di_y_d_tau_ref_y * vector_plane_lb[1] + di_z_d_tau_ref_y * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_y + tau_vector[1] * di_y_d_tau_ref_y +
                                        tau_vector[2] * di_z_d_tau_ref_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_y_d_tau_ref_z = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_z * vector_plane_lb[
            0] + di_y_d_tau_ref_z * vector_plane_lb[1] + di_z_d_tau_ref_z * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_z + tau_vector[1] * di_y_d_tau_ref_z +
                                        tau_vector[2] * di_z_d_tau_ref_z) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2

        par_vector = data_frame['vectors'][i] / np.linalg.norm(data_frame['vectors'][i])
        par = np.dot(pkmu_vector, par_vector) * np.array(pkmu_vector) / np.linalg.norm(pkmu_vector)

        dt_y_dp_k_mu_mom_x = (unit_l[1] / np.tan(angle)) * (2 * pkmu_vector[0] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[0] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[0] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[0] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[1] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_x +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_x +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_x) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_x + tau_vector[1] * di_y_dp_k_mu_mom_x + tau_vector[
            2] * di_z_dp_k_mu_mom_x) / np.linalg.norm(tau_vector)) / bottom_content ** 2 - \
                             ((pkmu_vector[1] * vector_plane_lb[0]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[1] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[0] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2

        dt_y_dp_k_mu_mom_y = (unit_l[1] / np.tan(angle)) * (2 * pkmu_vector[1] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[1] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[1] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[1] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[1] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_y +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_y +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_y) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_y + tau_vector[1] * di_y_dp_k_mu_mom_y + tau_vector[
            2] * di_z_dp_k_mu_mom_y) / np.linalg.norm(tau_vector)) / bottom_content ** 2 + \
                             1 - \
                             ((pkmu_vector[0] * vector_plane_lb[0] + 2 * pkmu_vector[1] * vector_plane_lb[1] +
                               pkmu_vector[2] * vector_plane_lb[2]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[1] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[1] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2
        dt_y_dp_k_mu_mom_z = (unit_l[1] / np.tan(angle)) * (2 * pkmu_vector[2] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[2] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[2] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[2] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[1] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_z +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_z +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_z) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_z + tau_vector[1] * di_y_dp_k_mu_mom_z + tau_vector[
            2] * di_z_dp_k_mu_mom_z) / np.linalg.norm(tau_vector)) / bottom_content ** 2 - \
                             ((pkmu_vector[1] * vector_plane_lb[2]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[1] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[2] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2

        dt_y_d_vector_lb_x = p_transverse / np.tan(angle) * vector_plane_lb[1] * (
                -vector_plane_lb[0] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[1]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (tau_vector[0] + vector_plane_lb[0] * di_x_d_vector_lb_x + vector_plane_lb[
                                         1] * di_y_d_vector_lb_x + vector_plane_lb[
                                          2] * di_z_d_vector_lb_x) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[0] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_x + tau_vector[
                                                 1] * di_y_d_vector_lb_x + tau_vector[
                                                         2] * di_z_d_vector_lb_x) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[1] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[0] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[0] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[0] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[0] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[1] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[0] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[0] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2

        dt_y_d_vector_lb_y = p_transverse / np.tan(angle) / np.linalg.norm(vector_plane_lb) + \
                             p_transverse / np.tan(angle) * vector_plane_lb[1] * (
                                     -vector_plane_lb[1] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[1]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (vector_plane_lb[0] * di_x_d_vector_lb_y + tau_vector[1] + vector_plane_lb[
                                         1] * di_y_d_vector_lb_y + vector_plane_lb[
                                          2] * di_z_d_vector_lb_y) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[1] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_y + tau_vector[
                                                 1] * di_y_d_vector_lb_y + tau_vector[
                                                         2] * di_z_d_vector_lb_y) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[1] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[1] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[1] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[1] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[1] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[1] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[1] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[1] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2
        dt_y_d_vector_lb_z = p_transverse / np.tan(angle) * vector_plane_lb[1] * (
                -vector_plane_lb[2] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[1]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (vector_plane_lb[0] * di_x_d_vector_lb_z + vector_plane_lb[
                                         1] * di_y_d_vector_lb_z + tau_vector[2] + vector_plane_lb[
                                          2] * di_z_d_vector_lb_z) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[2] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_z + tau_vector[
                                                 1] * di_y_d_vector_lb_z + tau_vector[
                                                         2] * di_z_d_vector_lb_z) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[1] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[2] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[2] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[2] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[2] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[1] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[2] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[2] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2

        dt_y_d_tau_mom_x = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * (
                                   (di_x_d_tau_mom_x * vector_plane_lb[0] + di_y_d_tau_mom_x * vector_plane_lb[
                                       1] + di_z_d_tau_mom_x *
                                    vector_plane_lb[2]) * bottom_content - np.dot(vector_plane_lb,
                                                                                  tau_vector) * np.linalg.norm(
                               vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_x + tau_vector[1] * di_y_d_tau_mom_x +
                                                   tau_vector[2] * di_z_d_tau_mom_x) / np.linalg.norm(
                               tau_vector)) / bottom_content ** 2
        dt_y_d_tau_mom_y = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_mom_y * vector_plane_lb[
            0] + di_y_d_tau_mom_y * vector_plane_lb[1] + di_z_d_tau_mom_y * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_y + tau_vector[1] * di_y_d_tau_mom_y +
                                        tau_vector[2] * di_z_d_tau_mom_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_y_d_tau_mom_z = p_transverse * unit_l[1] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_mom_z * vector_plane_lb[
            0] + di_y_d_tau_mom_z * vector_plane_lb[1] + di_z_d_tau_mom_z * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_z + tau_vector[1] * di_y_d_tau_mom_z +
                                        tau_vector[2] * di_z_d_tau_mom_z) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        sigma_tau_y = np.sqrt(dt_y_dp_k_mu_ref_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                              dt_y_dp_k_mu_ref_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                              dt_y_dp_k_mu_ref_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                              dt_y_d_tau_ref_x ** 2 * data_frame['tauMu_REFP_COVXX'][i] +
                              dt_y_d_tau_ref_y ** 2 * data_frame['tauMu_REFP_COVYY'][i] +
                              dt_y_d_tau_ref_z ** 2 * data_frame['tauMu_REFP_COVZZ'][i] +
                              dt_y_d_vector_lb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                              dt_y_d_vector_lb_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                              dt_y_d_vector_lb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                              dt_y_d_vector_lb_x ** 2 * data_frame['Lb_OWNPV_XERR'][i] ** 2 +
                              dt_y_d_vector_lb_y ** 2 * data_frame['Lb_OWNPV_YERR'][i] ** 2 +
                              dt_y_d_vector_lb_z ** 2 * data_frame['Lb_OWNPV_ZERR'][i] ** 2 +
                              dt_y_dp_k_mu_mom_x ** 2 * data_frame['pKmu_P_COVXX'][i] +
                              dt_y_dp_k_mu_mom_y ** 2 * data_frame['pKmu_P_COVYY'][i] +
                              dt_y_dp_k_mu_mom_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                              dt_y_d_tau_mom_x ** 2 * data_frame['tauMu_P_COVXX'][i] +
                              dt_y_d_tau_mom_y ** 2 * data_frame['tauMu_P_COVYY'][i] +
                              dt_y_d_tau_mom_z ** 2 * data_frame['tauMu_P_COVZZ'][i] +
                              2 * dt_y_dp_k_mu_mom_x * dt_y_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                              2 * dt_y_dp_k_mu_mom_x * dt_y_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_y_dp_k_mu_mom_x * dt_y_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_y_dp_k_mu_mom_y * dt_y_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                              2 * dt_y_dp_k_mu_mom_y * dt_y_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_y_dp_k_mu_mom_y * dt_y_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_y_dp_k_mu_mom_z * dt_y_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_y_dp_k_mu_mom_z * dt_y_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_y_dp_k_mu_mom_z * dt_y_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_y_d_tau_mom_x * dt_y_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PX_X'][i] +
                              2 * dt_y_d_tau_mom_x * dt_y_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_y_d_tau_mom_x * dt_y_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_y_d_tau_mom_y * dt_y_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PY_X'][i] +
                              2 * dt_y_d_tau_mom_y * dt_y_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_y_d_tau_mom_y * dt_y_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_y_d_tau_mom_z * dt_y_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_y_d_tau_mom_z * dt_y_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_y_d_tau_mom_z * dt_y_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                              2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                              2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXX'][i] +
                              2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVXY'][i] +
                              2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVXZ'][i] +
                              2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXY'][i] +
                              2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYY'][i] +
                              2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVYZ'][i] +
                              2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXZ'][i] +
                              2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYZ'][i] +
                              2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVZZ'][i])

        dt_z_dp_k_mu_ref_x = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * (((di_x_dp_k_mu_ref_x - 1) * vector_plane_lb[
            0] + di_y_dp_k_mu_ref_x * vector_plane_lb[1] + di_z_dp_k_mu_ref_x * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * (-1 + di_x_dp_k_mu_ref_x) + tau_vector[1] * di_y_dp_k_mu_ref_x +
                                        tau_vector[2] * di_z_dp_k_mu_ref_x) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_z_dp_k_mu_ref_y = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_dp_k_mu_ref_y * vector_plane_lb[
            0] + (di_y_dp_k_mu_ref_y - 1) * vector_plane_lb[1] + di_z_dp_k_mu_ref_y * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_ref_y + tau_vector[1] * (di_y_dp_k_mu_ref_y - 1) +
                                        tau_vector[2] * di_z_dp_k_mu_ref_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_z_dp_k_mu_ref_z = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_dp_k_mu_ref_z * vector_plane_lb[
            0] + di_y_dp_k_mu_ref_z * vector_plane_lb[1] + (di_z_dp_k_mu_ref_z - 1) * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_ref_z + tau_vector[1] * di_y_dp_k_mu_ref_z +
                                        tau_vector[2] * (di_z_dp_k_mu_ref_z - 1)) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2

        dt_z_d_tau_ref_x = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_x * vector_plane_lb[
            0] + di_y_d_tau_ref_x * vector_plane_lb[2] + di_z_d_tau_ref_x * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_x + tau_vector[1] * di_y_d_tau_ref_x +
                                        tau_vector[2] * di_z_d_tau_ref_x) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_z_d_tau_ref_y = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_y * vector_plane_lb[
            0] + di_y_d_tau_ref_y * vector_plane_lb[2] + di_z_d_tau_ref_y * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_y + tau_vector[1] * di_y_d_tau_ref_y +
                                        tau_vector[2] * di_z_d_tau_ref_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_z_d_tau_ref_z = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_ref_z * vector_plane_lb[
            0] + di_y_d_tau_ref_z * vector_plane_lb[2] + di_z_d_tau_ref_z * vector_plane_lb[
                                                             2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_ref_z + tau_vector[1] * di_y_d_tau_ref_z +
                                        tau_vector[2] * di_z_d_tau_ref_z) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2

        par_vector = data_frame['vectors'][i] / np.linalg.norm(data_frame['vectors'][i])
        par = np.dot(pkmu_vector, par_vector) * np.array(pkmu_vector) / np.linalg.norm(pkmu_vector)

        dt_z_dp_k_mu_mom_x = (unit_l[2] / np.tan(angle)) * (2 * pkmu_vector[0] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[0] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[0] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[0] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[2] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_x +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_x +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_x) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_x + tau_vector[1] * di_y_dp_k_mu_mom_x + tau_vector[
            2] * di_z_dp_k_mu_mom_x) / np.linalg.norm(tau_vector)) / bottom_content ** 2 - \
                             ((pkmu_vector[2] * vector_plane_lb[0]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[2] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[0] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2

        dt_z_dp_k_mu_mom_y = (unit_l[2] / np.tan(angle)) * (2 * pkmu_vector[1] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[1] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[1] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[1] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[2] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_y +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_y +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_y) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_y + tau_vector[1] * di_y_dp_k_mu_mom_y + tau_vector[
            2] * di_z_dp_k_mu_mom_y) / np.linalg.norm(tau_vector)) / bottom_content ** 2 - \
                             ((pkmu_vector[2] * vector_plane_lb[1]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[2] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[1] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2
        dt_z_dp_k_mu_mom_z = (unit_l[2] / np.tan(angle)) * (2 * pkmu_vector[2] - 2 / np.linalg.norm(vector_plane_lb) * (
                vector_plane_lb[2] * np.linalg.norm(pkmu_vector) + np.dot(vector_plane_lb, pkmu_vector) *
                pkmu_vector[2] / np.linalg.norm(pkmu_vector)) + 2 * np.dot(pkmu_vector, vector_plane_lb) *
                                                            vector_plane_lb[2] / np.linalg.norm(
                    vector_plane_lb) ** 2) / p_transverse * 0.5 + \
                             (unit_l[2] * p_transverse) / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * ((vector_plane_lb[0] * di_x_dp_k_mu_mom_z +
                                                                              vector_plane_lb[1] * di_y_dp_k_mu_mom_z +
                                                                              vector_plane_lb[
                                                                                  2] * di_z_dp_k_mu_mom_z) * np.linalg.norm(
            tau_vector) * np.linalg.norm(vector_plane_lb) - np.dot(tau_vector, vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) * (tau_vector[0] * di_x_dp_k_mu_mom_z + tau_vector[1] * di_y_dp_k_mu_mom_z + tau_vector[
            2] * di_z_dp_k_mu_mom_z) / np.linalg.norm(tau_vector)) / bottom_content ** 2 + \
                             1 - \
                             ((pkmu_vector[0] * vector_plane_lb[0] + pkmu_vector[1] * vector_plane_lb[1] + 2 *
                               pkmu_vector[2] * vector_plane_lb[2]) * np.linalg.norm(vector_plane_lb) * np.linalg.norm(
                                 pkmu_vector) - (np.dot(pkmu_vector, vector_plane_lb) * pkmu_vector[2] * np.linalg.norm(
                                 vector_plane_lb) * pkmu_vector[2] / np.linalg.norm(pkmu_vector))) / (
                                     np.linalg.norm(pkmu_vector) * np.linalg.norm(vector_plane_lb)) ** 2

        dt_z_d_vector_lb_x = p_transverse / np.tan(angle) * vector_plane_lb[2] * (
                -vector_plane_lb[0] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[2]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (tau_vector[0] + vector_plane_lb[0] * di_x_d_vector_lb_x + vector_plane_lb[
                                         1] * di_y_d_vector_lb_x + vector_plane_lb[
                                          2] * di_z_d_vector_lb_x) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[0] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_x + tau_vector[
                                                 1] * di_y_d_vector_lb_x + tau_vector[
                                                         2] * di_z_d_vector_lb_x) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[2] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[0] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[0] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[0] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[0] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[2] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[0] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[0] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2

        dt_z_d_vector_lb_y = p_transverse / np.tan(angle) * vector_plane_lb[2] * (
                                     -vector_plane_lb[1] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[2]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (vector_plane_lb[0] * di_x_d_vector_lb_y + tau_vector[1] + vector_plane_lb[
                                         1] * di_y_d_vector_lb_y + vector_plane_lb[
                                          2] * di_z_d_vector_lb_y) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[1] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_y + tau_vector[
                                                 1] * di_y_d_vector_lb_y + tau_vector[
                                                         2] * di_z_d_vector_lb_y) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[2] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[1] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[1] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[1] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[1] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[2] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[1] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[1] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2
        dt_z_d_vector_lb_z = p_transverse / np.tan(angle) / np.linalg.norm(vector_plane_lb) + \
            p_transverse / np.tan(angle) * vector_plane_lb[2] * (
                -vector_plane_lb[2] / np.linalg.norm(vector_plane_lb) ** 3) + \
                             (p_transverse * unit_l[2]) * 1 / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                                     1 / np.sqrt(1 - angle_content ** 2)) * (
                                     (vector_plane_lb[0] * di_x_d_vector_lb_z + vector_plane_lb[
                                         1] * di_y_d_vector_lb_z + tau_vector[2] + vector_plane_lb[
                                          2] * di_z_d_vector_lb_z) * bottom_content - np.dot(tau_vector,
                                                                                             vector_plane_lb) * (
                                             np.linalg.norm(
                                                 tau_vector) * vector_plane_lb[2] / np.linalg.norm(
                                         vector_plane_lb) + np.linalg.norm(vector_plane_lb) * (
                                                     tau_vector[0] * di_x_d_vector_lb_z + tau_vector[
                                                 1] * di_y_d_vector_lb_z + tau_vector[
                                                         2] * di_z_d_vector_lb_z) / np.linalg.norm(
                                         tau_vector))) / bottom_content ** 2 + \
                             (unit_l[2] / np.tan(angle)) * (-2 * np.linalg.norm(pkmu_vector) * (
                pkmu_vector[2] * np.linalg.norm(vector_plane_lb) - vector_plane_lb[2] * np.dot(pkmu_vector,
                                                                                               vector_plane_lb) / np.linalg.norm(
            vector_plane_lb)) / (np.linalg.norm(vector_plane_lb) ** 2) + (2 * pkmu_vector[2] * np.dot(pkmu_vector,
                                                                                                      vector_plane_lb) * np.linalg.norm(
            vector_plane_lb) ** 2 - 2 * vector_plane_lb[2] * (np.dot(pkmu_vector,
                                                                     vector_plane_lb)) ** 2) / np.linalg.norm(
            vector_plane_lb) ** 4) / p_transverse * 0.5 + \
                             0 - \
                             pkmu_vector[2] / np.linalg.norm(pkmu_vector) * \
                             (pkmu_vector[2] * np.linalg.norm(vector_plane_lb) - np.dot(vector_plane_lb, pkmu_vector) *
                              vector_plane_lb[2] / np.linalg.norm(vector_plane_lb)) / np.linalg.norm(
            vector_plane_lb) ** 2

        dt_z_d_tau_mom_x = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * (
                                   (di_x_d_tau_mom_x * vector_plane_lb[0] + di_y_d_tau_mom_x * vector_plane_lb[
                                       1] + di_z_d_tau_mom_x *
                                    vector_plane_lb[2]) * bottom_content - np.dot(vector_plane_lb,
                                                                                  tau_vector) * np.linalg.norm(
                               vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_x + tau_vector[1] * di_y_d_tau_mom_x +
                                                   tau_vector[2] * di_z_d_tau_mom_x) / np.linalg.norm(
                               tau_vector)) / bottom_content ** 2
        dt_z_d_tau_mom_y = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_mom_y * vector_plane_lb[
            0] + di_y_d_tau_mom_y * vector_plane_lb[1] + di_z_d_tau_mom_y * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_y + tau_vector[1] * di_y_d_tau_mom_y +
                                        tau_vector[2] * di_z_d_tau_mom_y) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        dt_z_d_tau_mom_z = p_transverse * unit_l[2] / (np.tan(angle)) ** 2 * 1 / (np.cos(angle)) ** 2 * (
                1 / np.sqrt(1 - angle_content ** 2)) * ((di_x_d_tau_mom_z * vector_plane_lb[
            0] + di_y_d_tau_mom_z * vector_plane_lb[1] + di_z_d_tau_mom_z * vector_plane_lb[2]) * bottom_content -
                                                        np.dot(vector_plane_lb, tau_vector) * np.linalg.norm(
                    vector_plane_lb) * (tau_vector[0] * di_x_d_tau_mom_z + tau_vector[1] * di_y_d_tau_mom_z +
                                        tau_vector[2] * di_z_d_tau_mom_z) / np.linalg.norm(
                    tau_vector)) / bottom_content ** 2
        sigma_tau_z = np.sqrt(dt_z_dp_k_mu_ref_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                              dt_z_dp_k_mu_ref_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                              dt_z_dp_k_mu_ref_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                              dt_z_d_tau_ref_x ** 2 * data_frame['tauMu_REFP_COVXX'][i] +
                              dt_z_d_tau_ref_y ** 2 * data_frame['tauMu_REFP_COVYY'][i] +
                              dt_z_d_tau_ref_z ** 2 * data_frame['tauMu_REFP_COVZZ'][i] +
                              dt_z_d_vector_lb_x ** 2 * data_frame['pKmu_REFP_COVXX'][i] +
                              dt_z_d_vector_lb_y ** 2 * data_frame['pKmu_REFP_COVYY'][i] +
                              dt_z_d_vector_lb_z ** 2 * data_frame['pKmu_REFP_COVZZ'][i] +
                              dt_z_d_vector_lb_x ** 2 * data_frame['Lb_OWNPV_XERR'][i] ** 2 +
                              dt_z_d_vector_lb_y ** 2 * data_frame['Lb_OWNPV_YERR'][i] ** 2 +
                              dt_z_d_vector_lb_z ** 2 * data_frame['Lb_OWNPV_ZERR'][i] ** 2 +
                              dt_z_dp_k_mu_mom_x ** 2 * data_frame['pKmu_P_COVXX'][i] +
                              dt_z_dp_k_mu_mom_y ** 2 * data_frame['pKmu_P_COVYY'][i] +
                              dt_z_dp_k_mu_mom_z ** 2 * data_frame['pKmu_P_COVZZ'][i] +
                              dt_z_d_tau_mom_x ** 2 * data_frame['tauMu_P_COVXX'][i] +
                              dt_z_d_tau_mom_y ** 2 * data_frame['tauMu_P_COVYY'][i] +
                              dt_z_d_tau_mom_z ** 2 * data_frame['tauMu_P_COVZZ'][i] +
                              2 * dt_z_dp_k_mu_mom_x * dt_z_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                              2 * dt_z_dp_k_mu_mom_x * dt_z_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_z_dp_k_mu_mom_x * dt_z_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_z_dp_k_mu_mom_y * dt_z_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                              2 * dt_z_dp_k_mu_mom_y * dt_z_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_z_dp_k_mu_mom_y * dt_z_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_z_dp_k_mu_mom_z * dt_z_dp_k_mu_ref_x * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_z_dp_k_mu_mom_z * dt_z_dp_k_mu_ref_y * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_z_dp_k_mu_mom_z * dt_z_dp_k_mu_ref_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_z_d_tau_mom_x * dt_z_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PX_X'][i] +
                              2 * dt_z_d_tau_mom_x * dt_z_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_z_d_tau_mom_x * dt_z_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_z_d_tau_mom_y * dt_z_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PY_X'][i] +
                              2 * dt_z_d_tau_mom_y * dt_z_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_z_d_tau_mom_y * dt_z_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_z_d_tau_mom_z * dt_z_d_tau_ref_x * data_frame['tauMu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_z_d_tau_mom_z * dt_z_d_tau_ref_y * data_frame['tauMu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_z_d_tau_mom_z * dt_z_d_tau_ref_z * data_frame['tauMu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_X'][i] +
                              2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_X'][i] +
                              2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_X'][i] +
                              2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Y'][i] +
                              2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Y'][i] +
                              2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Y'][i] +
                              2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_mom_x * data_frame['pKmu_P_REFP_COV_PX_Z'][i] +
                              2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_mom_y * data_frame['pKmu_P_REFP_COV_PY_Z'][i] +
                              2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_mom_z * data_frame['pKmu_P_REFP_COV_PZ_Z'][i] +
                              2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXX'][i] +
                              2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVXY'][i] +
                              2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVXZ'][i] +
                              2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXY'][i] +
                              2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYY'][i] +
                              2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVYZ'][i] +
                              2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_ref_x * data_frame['pKmu_REFP_COVXZ'][i] +
                              2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_ref_y * data_frame['pKmu_REFP_COVYZ'][i] +
                              2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_ref_z * data_frame['pKmu_REFP_COVZZ'][i])
        print(tau_mom, [sigma_tau_x, sigma_tau_y, sigma_tau_z])
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
    # TODO column with errors
    return data_frame


def mass(frame_array):
    return frame_array[0] ** 2 - frame_array[1] ** 2 + frame_array[2] ** 2 + frame_array[3] ** 2


def momentum(frame_array):
    mom = pd.concat([frame_array[1], frame_array[2], + frame_array[3]], axis=1)
    mom.columns = ['X', 'Y', 'Z']
    return mom


def plot_result(data_frame):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tau_P', 'tau']]
    particles = ['Kminus_P', 'proton_P', 'mu1_P', 'tau_P']
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    # TODO add errors for moms and energy
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    # TODO add error for sum_m
    plt.hist(sum_m, bins=50, range=[0, 50000])
    plt.xlabel('pkmutau mass')
    plt.ylabel('frequency')
    plt.show()
    return sum_m


def get_missing_mass(data_frame):
    particles_associations1 = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['mu1_P', 'mu']]
    particles1 = ['Kminus_P', 'proton_P', 'mu1_P']
    particles_associations2 = [['Kminus_P', 'K'], ['proton_P', 'proton'], ['tauMu_P', 'mu']]
    particles2 = ['Kminus_P', 'proton_P', 'tauMu_P']
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
    df = transverse_momentum(df, vec)
    df = line_plane_intersection(df)
    df = tau_momentum_mass(df)
    plot_result(df)
