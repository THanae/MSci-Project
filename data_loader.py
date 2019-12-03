import uproot
import numpy as np


def load_data(columns, year=2016):
    columns = set(columns)
    # file = uproot.open(f"merge_{year}.root")
    # file = uproot.open(f"lb_test.root")
    file = uproot.open(f"Lb2pKmumu_2018_MagUp.root")
    # events = file['LbTuple/DecayTree']
    events = file['DecayTree']
    print(events.show())
    events_frame = events.pandas.df(columns)
    print(events_frame.index)
    # events_frame = events_frame.reset_index(level=['subentry'])
    # no sub-entries when using ownpv so sub-entries have something to do with other primary vertices
    # events_frame = events_frame[events_frame['subentry'] < 1]
    # events_frame = events_frame.drop('subentry', axis=1)
    return events_frame


def add_branches():
    """
    Returns branches needed for analysis - for now returns branches used by Matt
    :return:
    """
    lb = ['Lb_M', 'Lb_ENDVERTEX_CHI2', 'Lb_DIRA_OWNPV', 'Lb_M', 'Lb_FD_OWNPV', 'Lb_OWNPV_X', 'Lb_OWNPV_Y', 'Lb_OWNPV_Z',
          'Lb_OWNPV_XERR', 'Lb_OWNPV_YERR', 'Lb_OWNPV_ZERR', 'Lb_ENDVERTEX_X', 'Lb_ENDVERTEX_Y', 'Lb_ENDVERTEX_Z',
          'Lb_ENDVERTEX_XERR', 'Lb_ENDVERTEX_YERR', 'Lb_ENDVERTEX_ZERR', "Lb_PE", "Lb_PX", "Lb_PY", "Lb_PZ", "Lb_P"]
    pkmu = ['pKmu_P', 'pKmu_ENDVERTEX_X', 'pKmu_ENDVERTEX_Y', 'pKmu_ENDVERTEX_Z', 'pKmu_OWNPV_X', 'pKmu_OWNPV_Y',
            'pKmu_OWNPV_Z', 'pKmu_PE', 'pKmu_PX', 'pKmu_PY', 'pKmu_PZ']
    cleaning = ["proton_P", "proton_PT", "Kminus_PT", "mu1_P", "mu1_PT", "tauMu_P", "tauMu_PT"]
    proton = ["proton_PE", "proton_PX", "proton_PY", "proton_PZ", "proton_P"]
    kminus = ["Kminus_PE", "Kminus_PX", "Kminus_PY", "Kminus_PZ", "Kminus_P"]
    mu1 = ["mu1_PE", "mu1_PX", "mu1_PY", "mu1_PZ", "mu1_P"]
    mu2 = ["tauMu_PE", "tauMu_PX", "tauMu_PY", "tauMu_PZ", "tauMu_P", 'tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']
    error_pkmu_ref = ['pKmu_REFP_COVXX', 'pKmu_REFP_COVYY', 'pKmu_REFP_COVZZ', 'pKmu_REFP_COVXY', 'pKmu_REFP_COVXZ',
                      'pKmu_REFP_COVYZ']
    error_taumu = ['tauMu_REFP_COVXX', 'tauMu_REFP_COVYY', 'tauMu_REFP_COVZZ', 'tauMu_REFP_COVXY', 'tauMu_REFP_COVXZ',
                   'tauMu_REFP_COVYZ']
    p_ref_cov = [x + '_P_REFP_COV_P' + letters for x in ['proton', 'Kminus', 'mu1', 'tauMu', 'Lb', 'pKmu'] for letters
                 in ['X_X', 'Y_X', 'Y_Y', 'Z_Z', 'Y_Z', 'Z_X', 'Z_Y', 'X_Y', 'X_Z', 'Y_Z']]
    p_cov = [x + '_P_COV' + letters for x in ['proton', 'Kminus', 'mu1', 'tauMu', 'Lb', 'pKmu'] for letters in
             ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']]
    errors = error_pkmu_ref + error_taumu + p_cov + p_ref_cov
    chi_squared = ['pKmu_OWNPV_CHI2', 'mu1_isMuon', 'pKmu_ENDVERTEX_CHI2']
    impact_parameter = ['proton_IPCHI2_OWNPV', 'Kminus_IPCHI2_OWNPV', 'mu1_IPCHI2_OWNPV', 'tauMu_IPCHI2_OWNPV',
                        'pKmu_IPCHI2_OWNPV']
    pid = ['proton_PIDe', 'Kminus_PIDe', 'Kminus_PIDK', 'Kminus_PIDmu', 'proton_PIDK', 'proton_PIDp', 'proton_PIDmu',
           'proton_PIDp', 'Kminus_PIDK', 'mu1_PIDmu', 'tauMu_PIDmu', 'proton_PIDK', 'proton_PIDmu',
           'Kminus_PIDp', 'Kminus_PIDmu', 'mu1_PIDp', 'mu1_PIDK', 'tauMu_PIDp', 'tauMu_PIDK']
    bdt = ['Lb_pmu_ISOLATION_BDT1']
    return lb + cleaning + proton + kminus + mu1 + mu2 + pkmu + errors + chi_squared + impact_parameter + pid + bdt


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
    check_for_both_charges(a)
    print(a)
