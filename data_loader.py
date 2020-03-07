import uproot
import numpy as np
import pandas as pd
from typing import List


def load_data(columns: List[str]):
    """
    Load data sets with the specified column names
    :param columns: columns to load
    :return:
    """
    columns = set(columns)
    file_names = ['Lb2pKmumu_2018_MagUp.root', 'Lb2pKmumu_2018_MagDown.root', 'Lb2pKmumu_2017_MagUp.root',
                  'Lb2pKmumu_2017_MagDown.root']
    # file_names = ['Lb2pKmumu_2017_MagDown.root', 'Lb2pKmumu_2018_MagDown.root']
    # file_names = ['Lb2pKmumu_MC_MagUp.root', 'Lb2pKmumu_MC_MagDown.root']
    # file_names = ['B2Ksttaumu_MC_MagUp.root', 'B2Ksttaumu_MC_MagDown.root']
    # file_names = ['B2Kstmutau_MC_MagUp.root', 'B2Kstmutau_MC_MagDown.root']
    events_frame = pd.DataFrame()
    for f in file_names:
        file = uproot.open(f'C:\\Users\\Hanae\\Documents\\MSci Project\\MsciCode\\{f}')
        events = file['DecayTree'] if 'B2' not in f else file['LbTuple/DecayTree']
        print(events.show())
        events_frame = pd.concat([events_frame, events.pandas.df(columns)], axis=0, ignore_index=True)
    print(events_frame.index)
    return events_frame


def add_branches():
    """
    Returns branches needed for analysis
    :return:
    """
    lb = ['Lb_M', 'Lb_ENDVERTEX_CHI2', 'Lb_DIRA_OWNPV', 'Lb_M', 'Lb_FD_OWNPV', 'Lb_OWNPV_X', 'Lb_OWNPV_Y', 'Lb_OWNPV_Z',
          'Lb_OWNPV_XERR', 'Lb_OWNPV_YERR', 'Lb_OWNPV_ZERR', 'Lb_ENDVERTEX_X', 'Lb_ENDVERTEX_Y', 'Lb_ENDVERTEX_Z',
          'Lb_ENDVERTEX_XERR', 'Lb_ENDVERTEX_YERR', 'Lb_ENDVERTEX_ZERR', 'Lb_PE', 'Lb_PX', 'Lb_PY', 'Lb_PZ', 'Lb_P',
          'Lb_L0Global_Dec', 'Lb_FDCHI2_OWNPV', 'Lb_ENDVERTEX_CHI2']
    lb_pmu = ['Lb_pmu_TR1_PIDp', 'Lb_pmu_TR1_PIDK', 'Lb_pmu_TR1_PIDmu', 'Lb_pmu_TR1_PIDpi']
    pkmu = ['pKmu_P', 'pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z', 'pKmu_OWNPV_X', 'pKmu_OWNPV_Y',
            'pKmu_OWNPV_Z', 'pKmu_PE', 'pKmu_PX', 'pKmu_PY', 'pKmu_PZ', 'pKmu_PT', 'pKmu_OWNPV_CHI2',
            'pKmu_ENDVERTEX_CHI2', 'pKmu_M']
    proton = ['proton_PE', 'proton_PX', 'proton_PY', 'proton_PZ', 'proton_P', 'proton_REFPX', 'proton_REFPY',
              'proton_REFPZ', 'proton_PT', 'proton_ProbNNp', 'proton_ProbNNe', 'proton_ProbNNk', 'proton_ProbNNpi',
              'proton_ProbNNmu', 'proton_ProbNNd', 'proton_ProbNNghost']
    kminus = ['Kminus_PE', 'Kminus_PX', 'Kminus_PY', 'Kminus_PZ', 'Kminus_P', 'Kminus_REFPX', 'Kminus_REFPY',
              'Kminus_REFPZ', 'Kminus_PT', 'Kminus_ProbNNk', 'Kminus_ProbNNe', 'Kminus_ProbNNp', 'Kminus_ProbNNpi',
              'Kminus_ProbNNmu', 'Kminus_ProbNNd', 'Kminus_ProbNNghost']
    mu1 = ['mu1_PE', 'mu1_PX', 'mu1_PY', 'mu1_PZ', 'mu1_P', 'mu1_REFPX', 'mu1_REFPY', 'mu1_REFPZ', 'mu1_L0Global_Dec',
           'mu1_PT', 'mu1_isMuon', 'mu1_ProbNNe', 'mu1_ProbNNk', 'mu1_ProbNNp', 'mu1_ProbNNpi', 'mu1_ProbNNmu',
           'mu1_ProbNNd', 'mu1_ProbNNghost']
    mu2 = ['tauMu_PE', 'tauMu_PX', 'tauMu_PY', 'tauMu_PZ', 'tauMu_P', 'tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ',
           'tauMu_L0Global_Dec', 'tauMu_PT', 'tauMu_isMuon', 'tauMu_ProbNNe', 'tauMu_ProbNNk', 'tauMu_ProbNNp',
           'tauMu_ProbNNpi', 'tauMu_ProbNNmu', 'tauMu_ProbNNghost']
    # error_pkmu_ref = ['pKmu_REFP_COVXX', 'pKmu_REFP_COVYY', 'pKmu_REFP_COVZZ', 'pKmu_REFP_COVXY', 'pKmu_REFP_COVXZ',
    #                   'pKmu_REFP_COVYZ']
    # error_taumu = ['tauMu_REFP_COVXX', 'tauMu_REFP_COVYY', 'tauMu_REFP_COVZZ', 'tauMu_REFP_COVXY', 'tauMu_REFP_COVXZ',
    #                'tauMu_REFP_COVYZ']
    # p_ref_cov = [x + '_P_REFP_COV_P' + letters for x in ['proton', 'Kminus', 'mu1', 'tauMu', 'Lb', 'pKmu'] for letters
    #              in ['X_X', 'Y_X', 'Y_Y', 'Z_Z', 'Y_Z', 'Z_X', 'Z_Y', 'X_Y', 'X_Z', 'Y_Z']]
    # p_cov = [x + '_P_COV' + letters for x in ['proton', 'Kminus', 'mu1', 'tauMu', 'Lb', 'pKmu'] for letters in
    #          ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']]
    # errors = error_pkmu_ref + error_taumu + p_cov + p_ref_cov
    errors = []
    ip = ['proton_IPCHI2_OWNPV', 'Kminus_IPCHI2_OWNPV', 'mu1_IPCHI2_OWNPV', 'tauMu_IPCHI2_OWNPV', 'pKmu_IPCHI2_OWNPV']
    pid = ['proton_PIDe', 'Kminus_PIDe', 'Kminus_PIDK', 'Kminus_PIDmu', 'proton_PIDK', 'proton_PIDp', 'proton_PIDmu',
           'proton_PIDp', 'Kminus_PIDK', 'mu1_PIDmu', 'tauMu_PIDmu', 'proton_PIDK', 'proton_PIDmu', 'proton_PIDd',
           'Kminus_PIDp', 'Kminus_PIDmu', 'mu1_PIDp', 'mu1_PIDK', 'tauMu_PIDp', 'tauMu_PIDK', 'Kminus_PIDd']
    isolation = ['Lb_pmu_ISOLATION_BDT1', 'Lb_pmu_ISOLATION_BDT2', 'Lb_pmu_ISOLATION_BDT3', 'Lb_pmu_ISOLATION_BDT4']
    mc = ['Lb_MC_MOTHER_ID', 'pKmu_MC_MOTHER_ID', 'proton_MC_MOTHER_ID', 'mu1_MC_MOTHER_ID', 'tauMu_MC_MOTHER_ID',
          'Kminus_MC_MOTHER_ID', 'Lb_MC_GD_MOTHER_ID', 'proton_MC_GD_MOTHER_ID', 'mu1_MC_GD_MOTHER_ID',
          'Kminus_MC_GD_MOTHER_ID', 'tauMu_MC_GD_MOTHER_ID', 'Kminus_ID', 'proton_ID', 'mu1_ID', 'tauMu_ID',
          'proton_MC_MOTHER_KEY', 'Kminus_MC_MOTHER_KEY', 'mu1_MC_MOTHER_KEY', 'tauMu_MC_MOTHER_KEY',
          'tauMu_MC_GD_MOTHER_KEY', 'Kminus_MC_GD_MOTHER_KEY', 'proton_MC_GD_MOTHER_KEY', 'mu1_MC_GD_MOTHER_KEY',
          'proton_MC_GD_GD_MOTHER_KEY', 'Kminus_MC_GD_GD_MOTHER_KEY', 'tauMu_MC_GD_GD_MOTHER_KEY', 'mu1_MC_GD_GD_MOTHER_KEY',
          'proton_TRUEID', 'Kminus_TRUEID', 'mu1_TRUEID', 'tauMu_TRUEID', 'proton_IPCHI2_OWNPV',
          'Kminus_TRACK_Type', 'proton_TRACK_Type', 'mu1_TRACK_Type', 'tauMu_TRACK_Type',
          'Kminus_TRACK_Key', 'proton_TRACK_Key', 'mu1_TRACK_Key', 'tauMu_TRACK_Key']
    mc = ['mu1_ID', 'tauMu_ID', 'Kminus_ID', 'proton_ID']
    # true = ['pKmu_TRUEENDVERTEX_X', 'pKmu_TRUEENDVERTEX_Y', 'pKmu_TRUEENDVERTEX_Z', 'tauMu_TRUEP_X', 'tauMu_TRUEP_Y',
    #         'tauMu_TRUEP_Z', 'pKmu_TRUEP_X', 'pKmu_TRUEP_Y', 'pKmu_TRUEP_Z']
    return lb + proton + kminus + mu1 + mu2 + pkmu + errors + ip + pid + isolation + mc + lb_pmu


def check_for_both_charges(data_frame):
    combinations = []
    for i in range(len(data_frame)):
        for j in range(len(data_frame)):
            if i == j:
                continue
            else:
                sigma = 0
                p_x_percentage = sigma * data_frame['proton_P_COVXX'][i] / data_frame['proton_PX'][i]
                px = (1 - p_x_percentage) * data_frame['proton_PX'][i] <= data_frame['proton_PX'][j] <= (
                        1 + p_x_percentage) * data_frame['proton_PX'][i]
                p_y_percentage = sigma * data_frame['proton_P_COVYY'][i] / data_frame['proton_PY'][i]
                py = (1 - p_y_percentage) * data_frame['proton_PY'][i] <= data_frame['proton_PY'][j] <= (
                        1 + p_y_percentage) * data_frame['proton_PY'][i]
                p_z_percentage = sigma * data_frame['proton_P_COVZZ'][i] / data_frame['proton_PZ'][i]
                pz = (1 - p_z_percentage) * data_frame['proton_PZ'][i] <= data_frame['proton_PZ'][j] <= (
                        1 + p_z_percentage) * data_frame['proton_PZ'][i]
                k_x_percentage = sigma * data_frame['Kminus_P_COVXX'][i] / data_frame['Kminus_PX'][i]
                kx = (1 - k_x_percentage) * data_frame['Kminus_PX'][i] <= data_frame['Kminus_PX'][j] <= (
                        1 + k_x_percentage) * data_frame['Kminus_PX'][i]
                k_y_percentage = sigma * data_frame['Kminus_P_COVYY'][i] / data_frame['Kminus_PY'][i]
                ky = (1 - k_y_percentage) * data_frame['Kminus_PY'][i] <= data_frame['Kminus_PY'][j] <= (
                        1 + k_y_percentage) * data_frame['Kminus_PY'][i]
                k_z_percentage = sigma * data_frame['Kminus_P_COVZZ'][i] / data_frame['Kminus_PZ'][i]
                kz = (1 - k_z_percentage) * data_frame['Kminus_PZ'][i] <= data_frame['Kminus_PZ'][j] <= (
                        1 + k_z_percentage) * data_frame['Kminus_PZ'][i]
                mu_x_percentage = sigma * data_frame['mu1_P_COVXX'][i] / data_frame['mu1_PX'][i]
                mux = (1 - mu_x_percentage) * data_frame['mu1_PX'][i] <= data_frame['tauMu_PX'][j] <= (
                        1 + mu_x_percentage) * data_frame['mu1_PX'][i]
                mu_y_percentage = sigma * data_frame['mu1_P_COVYY'][i] / data_frame['mu1_PY'][i]
                muy = (1 - mu_y_percentage) * data_frame['mu1_PY'][i] <= data_frame['tauMu_PY'][j] <= (
                        1 + mu_y_percentage) * data_frame['mu1_PY'][i]
                mu_z_percentage = sigma * data_frame['mu1_P_COVZZ'][i] / data_frame['tauMu_PZ'][i]
                muz = (1 - mu_z_percentage) * data_frame['mu1_PZ'][i] <= data_frame['tauMu_PZ'][j] <= (
                        1 + mu_z_percentage) * data_frame['mu1_PZ'][i]
                mux_dup = (1 - mu_x_percentage) * data_frame['mu1_PX'][i] <= data_frame['mu1_PX'][j] <= (
                        1 + mu_x_percentage) * data_frame['mu1_PX'][i]
                muy_dup = (1 - mu_y_percentage) * data_frame['mu1_PY'][i] <= data_frame['mu1_PY'][j] <= (
                        1 + mu_y_percentage) * data_frame['mu1_PY'][i]
                muz_dup = (1 - mu_z_percentage) * data_frame['mu1_PZ'][i] <= data_frame['mu1_PZ'][j] <= (
                        1 + mu_z_percentage) * data_frame['mu1_PZ'][i]
                tau_x_percentage = sigma * data_frame['tauMu_P_COVXX'][i] / data_frame['tauMu_PX'][i]
                taux = (1 - tau_x_percentage) * data_frame['tauMu_PX'][i] <= data_frame['tauMu_PX'][j] <= (
                        1 + tau_x_percentage) * data_frame['tauMu_PX'][i]
                tau_y_percentage = sigma * data_frame['tauMu_P_COVYY'][i] / data_frame['tauMu_PY'][i]
                tauy = (1 - tau_y_percentage) * data_frame['tauMu_PY'][i] <= data_frame['tauMu_PY'][j] <= (
                        1 + tau_y_percentage) * data_frame['tauMu_PY'][i]
                tau_z_percentage = sigma * data_frame['tauMu_P_COVZZ'][i] / data_frame['tauMu_PZ'][i]
                tauz = (1 - tau_z_percentage) * data_frame['tauMu_PZ'][i] <= data_frame['tauMu_PZ'][j] <= (
                        1 + tau_z_percentage) * data_frame['tauMu_PZ'][i]
                if px and py and pz and kx and ky and kz and mux and muy and muz:
                    print(i, j)
                    combinations.append([i, j])
                elif px and py and pz and kx and ky and kz and mux_dup and muy_dup and muz_dup and taux and tauy and tauz:
                    print('dups', i, j)
                elif px and py and pz and kx and ky and kz and mux_dup and muy_dup and muz_dup:
                    print('mu dups', i, j)
                elif px and py and pz and kx and ky and kz and taux and tauy and tauz:
                    print('tau dups', i, j)
    print(combinations)
    print(len(combinations))
    combinations = np.sort(combinations)
    print(np.unique(combinations, axis=0))


if __name__ == '__main__':
    a = load_data(add_branches())
    # check_for_both_charges(a)
