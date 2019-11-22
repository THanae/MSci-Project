from data_loader import load_data, add_branches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def error_propagation_intersection(data_frame, inverse_A, determinant, ordinate, i, possible_calculation_u):
    ts = data_frame.loc[i]
    vector_with_mu1 = [ts['pKmu_PX'], ts['pKmu_PY'], ts['pKmu_PZ']]
    momentum_tauMu = [ts['tauMu_PX'], ts['tauMu_PY'], ts['tauMu_PZ']]
    vector_plane_lb = ts['vectors']
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
    sigma_u = np.sqrt(du_dp_k_mu_ref_x ** 2 * ts['pKmu_REFP_COVXX'] +
                      du_dp_k_mu_ref_y ** 2 * ts['pKmu_REFP_COVYY'] +
                      du_dp_k_mu_ref_z ** 2 * ts['pKmu_REFP_COVZZ'] +
                      du_d_tau_ref_x ** 2 * ts['tauMu_REFP_COVXX'] +
                      du_d_tau_ref_y ** 2 * ts['tauMu_REFP_COVYY'] +
                      du_d_tau_ref_z ** 2 * ts['tauMu_REFP_COVZZ'] +
                      du_d_vector_lb_x ** 2 * ts['pKmu_REFP_COVXX'] +
                      du_d_vector_lb_y ** 2 * ts['pKmu_REFP_COVYY'] +
                      du_d_vector_lb_z ** 2 * ts['pKmu_REFP_COVZZ'] +
                      du_d_vector_lb_x ** 2 * ts['Lb_OWNPV_XERR'] ** 2 +
                      du_d_vector_lb_y ** 2 * ts['Lb_OWNPV_YERR'] ** 2 +
                      du_d_vector_lb_z ** 2 * ts['Lb_OWNPV_ZERR'] ** 2 +
                      du_dp_k_mu_mom_x ** 2 * ts['pKmu_P_COVXX'] +
                      du_dp_k_mu_mom_y ** 2 * ts['pKmu_P_COVYY'] +
                      du_dp_k_mu_mom_z ** 2 * ts['pKmu_P_COVZZ'] +
                      du_d_tau_mom_x ** 2 * ts['tauMu_P_COVXX'] +
                      du_d_tau_mom_y ** 2 * ts['tauMu_P_COVYY'] +
                      du_d_tau_mom_z ** 2 * ts['tauMu_P_COVZZ'] +
                      2 * du_dp_k_mu_mom_x * du_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PX_X'] +
                      2 * du_dp_k_mu_mom_x * du_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PX_Y'] +
                      2 * du_dp_k_mu_mom_x * du_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PX_Z'] +
                      2 * du_dp_k_mu_mom_y * du_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PY_X'] +
                      2 * du_dp_k_mu_mom_y * du_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PY_Y'] +
                      2 * du_dp_k_mu_mom_y * du_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PY_Z'] +
                      2 * du_dp_k_mu_mom_z * du_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PZ_X'] +
                      2 * du_dp_k_mu_mom_z * du_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PZ_Y'] +
                      2 * du_dp_k_mu_mom_z * du_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PZ_Z'] +
                      2 * du_d_tau_mom_x * du_d_tau_ref_x * ts['tauMu_P_REFP_COV_PX_X'] +
                      2 * du_d_tau_mom_x * du_d_tau_ref_y * ts['tauMu_P_REFP_COV_PX_Y'] +
                      2 * du_d_tau_mom_x * du_d_tau_ref_z * ts['tauMu_P_REFP_COV_PX_Z'] +
                      2 * du_d_tau_mom_y * du_d_tau_ref_x * ts['tauMu_P_REFP_COV_PY_X'] +
                      2 * du_d_tau_mom_y * du_d_tau_ref_y * ts['tauMu_P_REFP_COV_PY_Y'] +
                      2 * du_d_tau_mom_y * du_d_tau_ref_z * ts['tauMu_P_REFP_COV_PY_Z'] +
                      2 * du_d_tau_mom_z * du_d_tau_ref_x * ts['tauMu_P_REFP_COV_PZ_X'] +
                      2 * du_d_tau_mom_z * du_d_tau_ref_y * ts['tauMu_P_REFP_COV_PZ_Y'] +
                      2 * du_d_tau_mom_z * du_d_tau_ref_z * ts['tauMu_P_REFP_COV_PZ_Z'] +
                      2 * du_d_vector_lb_x * du_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_X'] +
                      2 * du_d_vector_lb_x * du_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_X'] +
                      2 * du_d_vector_lb_x * du_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_X'] +
                      2 * du_d_vector_lb_y * du_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_Y'] +
                      2 * du_d_vector_lb_y * du_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_Y'] +
                      2 * du_d_vector_lb_y * du_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_Y'] +
                      2 * du_d_vector_lb_z * du_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_Z'] +
                      2 * du_d_vector_lb_z * du_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_Z'] +
                      2 * du_d_vector_lb_z * du_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_Z'] +
                      2 * du_d_vector_lb_x * du_dp_k_mu_ref_x * ts['pKmu_REFP_COVXX'] +
                      2 * du_d_vector_lb_x * du_dp_k_mu_ref_y * ts['pKmu_REFP_COVXY'] +
                      2 * du_d_vector_lb_x * du_dp_k_mu_ref_z * ts['pKmu_REFP_COVXZ'] +
                      2 * du_d_vector_lb_y * du_dp_k_mu_ref_x * ts['pKmu_REFP_COVXY'] +
                      2 * du_d_vector_lb_y * du_dp_k_mu_ref_y * ts['pKmu_REFP_COVYY'] +
                      2 * du_d_vector_lb_y * du_dp_k_mu_ref_z * ts['pKmu_REFP_COVYZ'] +
                      2 * du_d_vector_lb_z * du_dp_k_mu_ref_x * ts['pKmu_REFP_COVXZ'] +
                      2 * du_d_vector_lb_z * du_dp_k_mu_ref_y * ts['pKmu_REFP_COVYZ'] +
                      2 * du_d_vector_lb_z * du_dp_k_mu_ref_z * ts['pKmu_REFP_COVZZ'])

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
    sigma_i_x = np.sqrt(di_dp_k_mu_ref_x ** 2 * ts['pKmu_REFP_COVXX'] +
                        di_dp_k_mu_ref_y ** 2 * ts['pKmu_REFP_COVYY'] +
                        di_dp_k_mu_ref_z ** 2 * ts['pKmu_REFP_COVZZ'] +
                        di_d_tau_ref_x ** 2 * ts['tauMu_REFP_COVXX'] +
                        di_d_tau_ref_y ** 2 * ts['tauMu_REFP_COVYY'] +
                        di_d_tau_ref_z ** 2 * ts['tauMu_REFP_COVZZ'] +
                        di_d_vector_lb_x ** 2 * ts['pKmu_REFP_COVXX'] +
                        di_d_vector_lb_y ** 2 * ts['pKmu_REFP_COVYY'] +
                        di_d_vector_lb_z ** 2 * ts['pKmu_REFP_COVZZ'] +
                        di_d_vector_lb_x ** 2 * ts['Lb_OWNPV_XERR'] ** 2 +
                        di_d_vector_lb_y ** 2 * ts['Lb_OWNPV_YERR'] ** 2 +
                        di_d_vector_lb_z ** 2 * ts['Lb_OWNPV_ZERR'] ** 2 +
                        di_dp_k_mu_mom_x ** 2 * ts['pKmu_P_COVXX'] +
                        di_dp_k_mu_mom_y ** 2 * ts['pKmu_P_COVYY'] +
                        di_dp_k_mu_mom_z ** 2 * ts['pKmu_P_COVZZ'] +
                        di_d_tau_mom_x ** 2 * ts['tauMu_P_COVXX'] +
                        di_d_tau_mom_y ** 2 * ts['tauMu_P_COVYY'] +
                        di_d_tau_mom_z ** 2 * ts['tauMu_P_COVZZ'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PX_X'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PX_Y'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PX_Z'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PY_X'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PY_Y'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PY_Z'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PZ_X'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PZ_Y'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PX_X'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PX_Y'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PX_Z'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PY_X'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PY_Y'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PY_Z'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PZ_X'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PZ_Y'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_X'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_X'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_X'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_Y'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_Y'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_Y'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_Z'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_Z'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXX'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVXY'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVXZ'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXY'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVYY'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVYZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVYZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVZZ'])

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
    sigma_i_y = np.sqrt(di_dp_k_mu_ref_x ** 2 * ts['pKmu_REFP_COVXX'] +
                        di_dp_k_mu_ref_y ** 2 * ts['pKmu_REFP_COVYY'] +
                        di_dp_k_mu_ref_z ** 2 * ts['pKmu_REFP_COVZZ'] +
                        di_d_tau_ref_x ** 2 * ts['tauMu_REFP_COVXX'] +
                        di_d_tau_ref_y ** 2 * ts['tauMu_REFP_COVYY'] +
                        di_d_tau_ref_z ** 2 * ts['tauMu_REFP_COVZZ'] +
                        di_d_vector_lb_x ** 2 * ts['pKmu_REFP_COVXX'] +
                        di_d_vector_lb_y ** 2 * ts['pKmu_REFP_COVYY'] +
                        di_d_vector_lb_z ** 2 * ts['pKmu_REFP_COVZZ'] +
                        di_d_vector_lb_x ** 2 * ts['Lb_OWNPV_XERR'] ** 2 +
                        di_d_vector_lb_y ** 2 * ts['Lb_OWNPV_YERR'] ** 2 +
                        di_d_vector_lb_z ** 2 * ts['Lb_OWNPV_ZERR'] ** 2 +
                        di_dp_k_mu_mom_x ** 2 * ts['pKmu_P_COVXX'] +
                        di_dp_k_mu_mom_y ** 2 * ts['pKmu_P_COVYY'] +
                        di_dp_k_mu_mom_z ** 2 * ts['pKmu_P_COVZZ'] +
                        di_d_tau_mom_x ** 2 * ts['tauMu_P_COVXX'] +
                        di_d_tau_mom_y ** 2 * ts['tauMu_P_COVYY'] +
                        di_d_tau_mom_z ** 2 * ts['tauMu_P_COVZZ'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PX_X'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PX_Y'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PX_Z'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PY_X'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PY_Y'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PY_Z'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PZ_X'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PZ_Y'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PX_X'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PX_Y'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PX_Z'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PY_X'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PY_Y'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PY_Z'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PZ_X'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PZ_Y'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_X'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_X'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_X'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_Y'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_Y'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_Y'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_Z'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_Z'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXX'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVXY'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVXZ'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXY'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVYY'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVYZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVYZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVZZ'])
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
    sigma_i_z = np.sqrt(di_dp_k_mu_ref_x ** 2 * ts['pKmu_REFP_COVXX'] +
                        di_dp_k_mu_ref_y ** 2 * ts['pKmu_REFP_COVYY'] +
                        di_dp_k_mu_ref_z ** 2 * ts['pKmu_REFP_COVZZ'] +
                        di_d_tau_ref_x ** 2 * ts['tauMu_REFP_COVXX'] +
                        di_d_tau_ref_y ** 2 * ts['tauMu_REFP_COVYY'] +
                        di_d_tau_ref_z ** 2 * ts['tauMu_REFP_COVZZ'] +
                        di_d_vector_lb_x ** 2 * ts['pKmu_REFP_COVXX'] +
                        di_d_vector_lb_y ** 2 * ts['pKmu_REFP_COVYY'] +
                        di_d_vector_lb_z ** 2 * ts['pKmu_REFP_COVZZ'] +
                        di_d_vector_lb_x ** 2 * ts['Lb_OWNPV_XERR'] ** 2 +
                        di_d_vector_lb_y ** 2 * ts['Lb_OWNPV_YERR'] ** 2 +
                        di_d_vector_lb_z ** 2 * ts['Lb_OWNPV_ZERR'] ** 2 +
                        di_dp_k_mu_mom_x ** 2 * ts['pKmu_P_COVXX'] +
                        di_dp_k_mu_mom_y ** 2 * ts['pKmu_P_COVYY'] +
                        di_dp_k_mu_mom_z ** 2 * ts['pKmu_P_COVZZ'] +
                        di_d_tau_mom_x ** 2 * ts['tauMu_P_COVXX'] +
                        di_d_tau_mom_y ** 2 * ts['tauMu_P_COVYY'] +
                        di_d_tau_mom_z ** 2 * ts['tauMu_P_COVZZ'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PX_X'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PX_Y'] +
                        2 * di_dp_k_mu_mom_x * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PX_Z'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PY_X'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PY_Y'] +
                        2 * di_dp_k_mu_mom_y * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PY_Z'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_x * ts['pKmu_P_REFP_COV_PZ_X'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_y * ts['pKmu_P_REFP_COV_PZ_Y'] +
                        2 * di_dp_k_mu_mom_z * di_dp_k_mu_ref_z * ts['pKmu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PX_X'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PX_Y'] +
                        2 * di_d_tau_mom_x * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PX_Z'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PY_X'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PY_Y'] +
                        2 * di_d_tau_mom_y * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PY_Z'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_x * ts['tauMu_P_REFP_COV_PZ_X'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_y * ts['tauMu_P_REFP_COV_PZ_Y'] +
                        2 * di_d_tau_mom_z * di_d_tau_ref_z * ts['tauMu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_X'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_X'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_X'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_Y'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_Y'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_Y'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_x * ts['pKmu_P_REFP_COV_PX_Z'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_y * ts['pKmu_P_REFP_COV_PY_Z'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_mom_z * ts['pKmu_P_REFP_COV_PZ_Z'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXX'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVXY'] +
                        2 * di_d_vector_lb_x * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVXZ'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXY'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVYY'] +
                        2 * di_d_vector_lb_y * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVYZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_x * ts['pKmu_REFP_COVXZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_y * ts['pKmu_REFP_COVYZ'] +
                        2 * di_d_vector_lb_z * di_dp_k_mu_ref_z * ts['pKmu_REFP_COVZZ'])
    return sigma_i_x, sigma_i_y, sigma_i_z


def error_propagation_transverse_momentum(data_frame, vectors, i, pkmu_vector):
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
    return t_mom_err


def error_propagation_momentum_mass(data_frame, inverse_A, determinant, i, angle, unit_l, p_transverse, bottom_content,
                                    angle_content, possible_calculation_u, ordinate):
    temp_series = data_frame.loc[i]
    end_xyz = [temp_series['pKmu_ENDVERTEX_X'], temp_series['pKmu_ENDVERTEX_Y'], temp_series['pKmu_ENDVERTEX_Z']]
    tau_vector = temp_series['tau_decay_point'] - end_xyz
    momentum_tauMu = [temp_series['tauMu_PX'], temp_series['tauMu_PY'], temp_series['tauMu_PZ']]
    pkmu_vector = [temp_series['pKmu_PX'], temp_series['pKmu_PY'], temp_series['pKmu_PZ']]
    vector_plane_lb = temp_series['vectors']
    vector_with_mu1 = [temp_series['pKmu_PX'], temp_series['pKmu_PY'], temp_series['pKmu_PZ']]
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
    sigma_tau_x = np.sqrt(dt_x_dp_k_mu_ref_x ** 2 * temp_series['pKmu_REFP_COVXX'] +
                          dt_x_dp_k_mu_ref_y ** 2 * temp_series['pKmu_REFP_COVYY'] +
                          dt_x_dp_k_mu_ref_z ** 2 * temp_series['pKmu_REFP_COVZZ'] +
                          dt_x_d_tau_ref_x ** 2 * temp_series['tauMu_REFP_COVXX'] +
                          dt_x_d_tau_ref_y ** 2 * temp_series['tauMu_REFP_COVYY'] +
                          dt_x_d_tau_ref_z ** 2 * temp_series['tauMu_REFP_COVZZ'] +
                          dt_x_d_vector_lb_x ** 2 * temp_series['pKmu_REFP_COVXX'] +
                          dt_x_d_vector_lb_y ** 2 * temp_series['pKmu_REFP_COVYY'] +
                          dt_x_d_vector_lb_z ** 2 * temp_series['pKmu_REFP_COVZZ'] +
                          dt_x_d_vector_lb_x ** 2 * temp_series['Lb_OWNPV_XERR'] ** 2 +
                          dt_x_d_vector_lb_y ** 2 * temp_series['Lb_OWNPV_YERR'] ** 2 +
                          dt_x_d_vector_lb_z ** 2 * temp_series['Lb_OWNPV_ZERR'] ** 2 +
                          dt_x_dp_k_mu_mom_x ** 2 * temp_series['pKmu_P_COVXX'] +
                          dt_x_dp_k_mu_mom_y ** 2 * temp_series['pKmu_P_COVYY'] +
                          dt_x_dp_k_mu_mom_z ** 2 * temp_series['pKmu_P_COVZZ'] +
                          dt_x_d_tau_mom_x ** 2 * temp_series['tauMu_P_COVXX'] +
                          dt_x_d_tau_mom_y ** 2 * temp_series['tauMu_P_COVYY'] +
                          dt_x_d_tau_mom_z ** 2 * temp_series['tauMu_P_COVZZ'] +
                          2 * dt_x_dp_k_mu_mom_x * dt_x_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PX_X'] +
                          2 * dt_x_dp_k_mu_mom_x * dt_x_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PX_Y'] +
                          2 * dt_x_dp_k_mu_mom_x * dt_x_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PX_Z'] +
                          2 * dt_x_dp_k_mu_mom_y * dt_x_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PY_X'] +
                          2 * dt_x_dp_k_mu_mom_y * dt_x_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PY_Y'] +
                          2 * dt_x_dp_k_mu_mom_y * dt_x_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PY_Z'] +
                          2 * dt_x_dp_k_mu_mom_z * dt_x_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PZ_X'] +
                          2 * dt_x_dp_k_mu_mom_z * dt_x_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PZ_Y'] +
                          2 * dt_x_dp_k_mu_mom_z * dt_x_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PZ_Z'] +
                          2 * dt_x_d_tau_mom_x * dt_x_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PX_X'] +
                          2 * dt_x_d_tau_mom_x * dt_x_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PX_Y'] +
                          2 * dt_x_d_tau_mom_x * dt_x_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PX_Z'] +
                          2 * dt_x_d_tau_mom_y * dt_x_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PY_X'] +
                          2 * dt_x_d_tau_mom_y * dt_x_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PY_Y'] +
                          2 * dt_x_d_tau_mom_y * dt_x_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PY_Z'] +
                          2 * dt_x_d_tau_mom_z * dt_x_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PZ_X'] +
                          2 * dt_x_d_tau_mom_z * dt_x_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PZ_Y'] +
                          2 * dt_x_d_tau_mom_z * dt_x_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PZ_Z'] +
                          2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_X'] +
                          2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_X'] +
                          2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_X'] +
                          2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_Y'] +
                          2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_Y'] +
                          2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_Y'] +
                          2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_Z'] +
                          2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_Z'] +
                          2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_Z'] +
                          2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXX'] +
                          2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVXY'] +
                          2 * dt_x_d_vector_lb_x * dt_x_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVXZ'] +
                          2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXY'] +
                          2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVYY'] +
                          2 * dt_x_d_vector_lb_y * dt_x_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVYZ'] +
                          2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXZ'] +
                          2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVYZ'] +
                          2 * dt_x_d_vector_lb_z * dt_x_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVZZ'])

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
    sigma_tau_y = np.sqrt(dt_y_dp_k_mu_ref_x ** 2 * temp_series['pKmu_REFP_COVXX'] +
                          dt_y_dp_k_mu_ref_y ** 2 * temp_series['pKmu_REFP_COVYY'] +
                          dt_y_dp_k_mu_ref_z ** 2 * temp_series['pKmu_REFP_COVZZ'] +
                          dt_y_d_tau_ref_x ** 2 * temp_series['tauMu_REFP_COVXX'] +
                          dt_y_d_tau_ref_y ** 2 * temp_series['tauMu_REFP_COVYY'] +
                          dt_y_d_tau_ref_z ** 2 * temp_series['tauMu_REFP_COVZZ'] +
                          dt_y_d_vector_lb_x ** 2 * temp_series['pKmu_REFP_COVXX'] +
                          dt_y_d_vector_lb_y ** 2 * temp_series['pKmu_REFP_COVYY'] +
                          dt_y_d_vector_lb_z ** 2 * temp_series['pKmu_REFP_COVZZ'] +
                          dt_y_d_vector_lb_x ** 2 * temp_series['Lb_OWNPV_XERR'] ** 2 +
                          dt_y_d_vector_lb_y ** 2 * temp_series['Lb_OWNPV_YERR'] ** 2 +
                          dt_y_d_vector_lb_z ** 2 * temp_series['Lb_OWNPV_ZERR'] ** 2 +
                          dt_y_dp_k_mu_mom_x ** 2 * temp_series['pKmu_P_COVXX'] +
                          dt_y_dp_k_mu_mom_y ** 2 * temp_series['pKmu_P_COVYY'] +
                          dt_y_dp_k_mu_mom_z ** 2 * temp_series['pKmu_P_COVZZ'] +
                          dt_y_d_tau_mom_x ** 2 * temp_series['tauMu_P_COVXX'] +
                          dt_y_d_tau_mom_y ** 2 * temp_series['tauMu_P_COVYY'] +
                          dt_y_d_tau_mom_z ** 2 * temp_series['tauMu_P_COVZZ'] +
                          2 * dt_y_dp_k_mu_mom_x * dt_y_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PX_X'] +
                          2 * dt_y_dp_k_mu_mom_x * dt_y_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PX_Y'] +
                          2 * dt_y_dp_k_mu_mom_x * dt_y_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PX_Z'] +
                          2 * dt_y_dp_k_mu_mom_y * dt_y_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PY_X'] +
                          2 * dt_y_dp_k_mu_mom_y * dt_y_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PY_Y'] +
                          2 * dt_y_dp_k_mu_mom_y * dt_y_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PY_Z'] +
                          2 * dt_y_dp_k_mu_mom_z * dt_y_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PZ_X'] +
                          2 * dt_y_dp_k_mu_mom_z * dt_y_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PZ_Y'] +
                          2 * dt_y_dp_k_mu_mom_z * dt_y_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PZ_Z'] +
                          2 * dt_y_d_tau_mom_x * dt_y_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PX_X'] +
                          2 * dt_y_d_tau_mom_x * dt_y_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PX_Y'] +
                          2 * dt_y_d_tau_mom_x * dt_y_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PX_Z'] +
                          2 * dt_y_d_tau_mom_y * dt_y_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PY_X'] +
                          2 * dt_y_d_tau_mom_y * dt_y_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PY_Y'] +
                          2 * dt_y_d_tau_mom_y * dt_y_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PY_Z'] +
                          2 * dt_y_d_tau_mom_z * dt_y_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PZ_X'] +
                          2 * dt_y_d_tau_mom_z * dt_y_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PZ_Y'] +
                          2 * dt_y_d_tau_mom_z * dt_y_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PZ_Z'] +
                          2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_X'] +
                          2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_X'] +
                          2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_X'] +
                          2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_Y'] +
                          2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_Y'] +
                          2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_Y'] +
                          2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_Z'] +
                          2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_Z'] +
                          2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_Z'] +
                          2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXX'] +
                          2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVXY'] +
                          2 * dt_y_d_vector_lb_x * dt_y_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVXZ'] +
                          2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXY'] +
                          2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVYY'] +
                          2 * dt_y_d_vector_lb_y * dt_y_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVYZ'] +
                          2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXZ'] +
                          2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVYZ'] +
                          2 * dt_y_d_vector_lb_z * dt_y_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVZZ'])

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
    sigma_tau_z = np.sqrt(dt_z_dp_k_mu_ref_x ** 2 * temp_series['pKmu_REFP_COVXX'] +
                          dt_z_dp_k_mu_ref_y ** 2 * temp_series['pKmu_REFP_COVYY'] +
                          dt_z_dp_k_mu_ref_z ** 2 * temp_series['pKmu_REFP_COVZZ'] +
                          dt_z_d_tau_ref_x ** 2 * temp_series['tauMu_REFP_COVXX'] +
                          dt_z_d_tau_ref_y ** 2 * temp_series['tauMu_REFP_COVYY'] +
                          dt_z_d_tau_ref_z ** 2 * temp_series['tauMu_REFP_COVZZ'] +
                          dt_z_d_vector_lb_x ** 2 * temp_series['pKmu_REFP_COVXX'] +
                          dt_z_d_vector_lb_y ** 2 * temp_series['pKmu_REFP_COVYY'] +
                          dt_z_d_vector_lb_z ** 2 * temp_series['pKmu_REFP_COVZZ'] +
                          dt_z_d_vector_lb_x ** 2 * temp_series['Lb_OWNPV_XERR'] ** 2 +
                          dt_z_d_vector_lb_y ** 2 * temp_series['Lb_OWNPV_YERR'] ** 2 +
                          dt_z_d_vector_lb_z ** 2 * temp_series['Lb_OWNPV_ZERR'] ** 2 +
                          dt_z_dp_k_mu_mom_x ** 2 * temp_series['pKmu_P_COVXX'] +
                          dt_z_dp_k_mu_mom_y ** 2 * temp_series['pKmu_P_COVYY'] +
                          dt_z_dp_k_mu_mom_z ** 2 * temp_series['pKmu_P_COVZZ'] +
                          dt_z_d_tau_mom_x ** 2 * temp_series['tauMu_P_COVXX'] +
                          dt_z_d_tau_mom_y ** 2 * temp_series['tauMu_P_COVYY'] +
                          dt_z_d_tau_mom_z ** 2 * temp_series['tauMu_P_COVZZ'] +
                          2 * dt_z_dp_k_mu_mom_x * dt_z_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PX_X'] +
                          2 * dt_z_dp_k_mu_mom_x * dt_z_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PX_Y'] +
                          2 * dt_z_dp_k_mu_mom_x * dt_z_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PX_Z'] +
                          2 * dt_z_dp_k_mu_mom_y * dt_z_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PY_X'] +
                          2 * dt_z_dp_k_mu_mom_y * dt_z_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PY_Y'] +
                          2 * dt_z_dp_k_mu_mom_y * dt_z_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PY_Z'] +
                          2 * dt_z_dp_k_mu_mom_z * dt_z_dp_k_mu_ref_x * temp_series['pKmu_P_REFP_COV_PZ_X'] +
                          2 * dt_z_dp_k_mu_mom_z * dt_z_dp_k_mu_ref_y * temp_series['pKmu_P_REFP_COV_PZ_Y'] +
                          2 * dt_z_dp_k_mu_mom_z * dt_z_dp_k_mu_ref_z * temp_series['pKmu_P_REFP_COV_PZ_Z'] +
                          2 * dt_z_d_tau_mom_x * dt_z_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PX_X'] +
                          2 * dt_z_d_tau_mom_x * dt_z_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PX_Y'] +
                          2 * dt_z_d_tau_mom_x * dt_z_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PX_Z'] +
                          2 * dt_z_d_tau_mom_y * dt_z_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PY_X'] +
                          2 * dt_z_d_tau_mom_y * dt_z_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PY_Y'] +
                          2 * dt_z_d_tau_mom_y * dt_z_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PY_Z'] +
                          2 * dt_z_d_tau_mom_z * dt_z_d_tau_ref_x * temp_series['tauMu_P_REFP_COV_PZ_X'] +
                          2 * dt_z_d_tau_mom_z * dt_z_d_tau_ref_y * temp_series['tauMu_P_REFP_COV_PZ_Y'] +
                          2 * dt_z_d_tau_mom_z * dt_z_d_tau_ref_z * temp_series['tauMu_P_REFP_COV_PZ_Z'] +
                          2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_X'] +
                          2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_X'] +
                          2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_X'] +
                          2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_Y'] +
                          2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_Y'] +
                          2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_Y'] +
                          2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_mom_x * temp_series['pKmu_P_REFP_COV_PX_Z'] +
                          2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_mom_y * temp_series['pKmu_P_REFP_COV_PY_Z'] +
                          2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_mom_z * temp_series['pKmu_P_REFP_COV_PZ_Z'] +
                          2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXX'] +
                          2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVXY'] +
                          2 * dt_z_d_vector_lb_x * dt_z_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVXZ'] +
                          2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXY'] +
                          2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVYY'] +
                          2 * dt_z_d_vector_lb_y * dt_z_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVYZ'] +
                          2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_ref_x * temp_series['pKmu_REFP_COVXZ'] +
                          2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_ref_y * temp_series['pKmu_REFP_COVYZ'] +
                          2 * dt_z_d_vector_lb_z * dt_z_dp_k_mu_ref_z * temp_series['pKmu_REFP_COVZZ'])
    return [sigma_tau_x, sigma_tau_y, sigma_tau_z]
